"""kunlun_communicator"""

import logging
from contextlib import contextmanager
from torch.distributed import ProcessGroup
import torch
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

logger = logging.getLogger("vllm_kunlun")


class KunlunCommunicator(CudaCommunicator):
    """KunlunCommunicator"""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        """
            Initializes the CUDA Communicator.

        Args:
            cpu_group (ProcessGroup): The CPU process group.
            device (Optional[torch.device], optional): The device to use. Defaults to None.
            device_group (Optional[ProcessGroup], optional): The device process group. Defaults to None.
            unique_name (str, optional): The unique name of this communicator. Defaults to "".

        Raises:
            ValueError: If both ``device`` and ``device_group`` are not specified.
        """
        DeviceCommunicatorBase.__init__(
            self,
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all=use_all2all,
        )
        # CudaCommunicator.__init__ is intentionally skipped: pynccl, custom
        # all-reduce, quick-reduce and symmetric memory do not apply here. Its
        # methods are still inherited though, so define every attribute they
        # test for. Without this, DP hits
        # "'KunlunCommunicator' object has no attribute 'pynccl_comm'".
        self.use_custom_allreduce = False
        self.use_torch_symm_mem = False
        self.use_flashinfer_allreduce = False
        self.use_aiter_allreduce = False
        self.pynccl_comm = None
        self.qr_comm = None
        self.symm_mem_comm = None
        self.fi_ar_comm = None
        self.aiter_ar_comm = None
        self.ca_comm = None
        self.disabled = False
        with torch.cuda.device(device):
            self.stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            self.stream.synchronize()
            del data

        if self.use_all2all:
            self._init_all2all_manager(tcp_store_group)

    def _init_all2all_manager(self, tcp_store_group) -> None:
        """Create the all2all manager used by expert parallelism.

        This class deliberately skips ``CudaCommunicator.__init__`` (pynccl and
        custom all-reduce do not apply here), which also skips the block that
        builds ``all2all_manager``. Without it ``GPUModelRunner.__init__``
        trips ``assert all2all_manager is not None`` inside
        ``get_ep_all2all_manager()`` as soon as data parallelism is enabled.
        """
        from vllm.distributed.device_communicators import all2all

        backend = self.all2all_backend
        if backend in ("naive", "allgather_reducescatter"):
            cls = all2all.AgRsAll2AllManager
        elif backend == "deepep_high_throughput":
            cls = all2all.DeepEPHTAll2AllManager
        elif backend == "deepep_low_latency":
            cls = all2all.DeepEPLLAll2AllManager
        else:
            raise NotImplementedError(
                "all2all backend %r is not adapted for Kunlun; use "
                "allgather_reducescatter, deepep_high_throughput or "
                "deepep_low_latency" % (backend,)
            )
        self.all2all_manager = cls(self.cpu_group, tcp_store_group)
        logger.info("[KunlunPlugin] all2all manager: %s", cls.__name__)

    def all_reduce(self, input_):
        """all_reduce"""
        return DeviceCommunicatorBase.all_reduce(self, input_)

    def all_gather(self, input_, dim):
        """all_gather"""
        return DeviceCommunicatorBase.all_gather(self, input_, dim)

    def reduce_scatter(self, input_, dim: int = -1):
        """reduce_scatter over torch.distributed instead of pynccl.

        ``CudaCommunicator.reduce_scatter`` asserts ``pynccl_comm is not None``,
        which never holds here. Nothing exercised it until sequence-parallel MoE
        (tensor_parallel_size > 1 together with data_parallel_size > 1) started
        calling it from ``sp_reduce_scatter``.
        """
        return DeviceCommunicatorBase.reduce_scatter(self, input_, dim)

    def all_gatherv(self, input_, dim: int = 0, sizes=None):
        """Variable-size all-gather along dim 0.

        CudaCommunicator implements this on top of pynccl, which is not
        available here, so the AgRs all2all backend used by expert parallelism
        would trip ``assert pynccl_comm is not None``.

        BKCL/NCCL collectives require identical shapes on every rank, so the
        ragged case pads each rank's slice up to ``max(sizes)``, gathers the
        padded buffers and then drops the padding. Passing differently shaped
        tensors to ``torch.distributed.all_gather`` silently produces garbage.
        """
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        rank = self.rank_in_group
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _single(inp: torch.Tensor) -> torch.Tensor:
            inp = inp.contiguous()
            tail = tuple(inp.shape[1:])
            if sizes is None:
                out = torch.empty(
                    (inp.shape[0] * world_size,) + tail,
                    dtype=inp.dtype,
                    device=inp.device,
                )
                torch.distributed.all_gather_into_tensor(
                    out, inp, group=self.device_group
                )
                return out

            assert len(sizes) == world_size, "%d != %d" % (len(sizes), world_size)
            assert inp.shape[0] == sizes[rank], "%d != %d" % (
                inp.shape[0],
                sizes[rank],
            )
            pad_len = max(sizes)
            padded = torch.zeros(
                (pad_len,) + tail, dtype=inp.dtype, device=inp.device
            )
            padded[: sizes[rank]] = inp
            gathered = torch.empty(
                (pad_len * world_size,) + tail,
                dtype=inp.dtype,
                device=inp.device,
            )
            torch.distributed.all_gather_into_tensor(
                gathered, padded, group=self.device_group
            )
            return torch.cat(
                [
                    gathered[i * pad_len : i * pad_len + sizes[i]]
                    for i in range(world_size)
                ],
                dim=0,
            )

        if isinstance(input_, torch.Tensor):
            return _single(input_)
        return [_single(x) for x in input_]

    def reduce_scatterv(self, input_, dim: int = -1, sizes=None):
        """Variable-size reduce-scatter, counterpart of all_gatherv.

        Same shape restriction as above: the ragged case is padded with zeros
        (the identity for summation) so a fixed-shape reduce_scatter can be
        used, then the padding is dropped.
        """
        world_size = self.world_size
        rank = self.rank_in_group
        if dim < 0:
            dim += input_.dim()
        input_tensor = input_.movedim(0, dim).contiguous()
        tail = tuple(input_tensor.shape[1:])
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        if sizes is None:
            assert input_tensor.shape[0] % world_size == 0
            chunk = input_tensor.shape[0] // world_size
            output = torch.empty(
                (chunk,) + tail,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            torch.distributed.reduce_scatter_tensor(
                output, input_tensor, group=self.device_group
            )
            return output.movedim(0, dim).contiguous()

        assert len(sizes) == world_size, "%d != %d" % (len(sizes), world_size)
        assert input_tensor.shape[0] == sum(sizes)
        pad_len = max(sizes)
        padded = torch.zeros(
            (pad_len * world_size,) + tail,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        off = 0
        for i, n in enumerate(sizes):
            padded[i * pad_len : i * pad_len + n] = input_tensor[off : off + n]
            off += n
        reduced = torch.empty(
            (pad_len,) + tail,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        torch.distributed.reduce_scatter_tensor(
            reduced, padded, group=self.device_group
        )
        return reduced[: sizes[rank]].movedim(0, dim).contiguous()

    def gather(self, input_, dst, dim):
        """gather"""
        return DeviceCommunicatorBase.gather(self, input_, dst, dim)

    def send(self, tensor, dst):
        """send"""
        DeviceCommunicatorBase.send(self, tensor, dst)

    def recv(self, size, dtype, src):
        """recv"""
        return DeviceCommunicatorBase.recv(self, size, dtype, src)

    def destroy(self):
        """destroy"""
        pass

    @contextmanager
    def change_state(self, enable, stream):
        """
        A context manager to change the state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        if stream is None:
            stream = self.stream

        old_disable = self.disabled
        old_stream = self.stream

        self.stream = stream
        self.disabled = not enable
        yield

        self.disabled = old_disable
        self.stream = old_stream
