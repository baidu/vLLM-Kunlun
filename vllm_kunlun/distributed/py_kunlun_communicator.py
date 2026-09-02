"""XCCL weight-sync communicator for KunLun XPU (single-node and multi-node).

Env switches:
  ``XCCL_WEIGHT_SYNC=0``    fall back to the original TCPStore path (bisecting)
  ``XCCL_FORCE_FLAT_PG=1``  build one flat group even across hosts (debug only)
"""

import atexit
import io
import os
import socket
from datetime import timedelta
from typing import TYPE_CHECKING, Union

import torch
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed import PrefixStore

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup

# Backend registered by torch_xmlir for the KunLun XCCL/BKCL library.
_XCCL_BACKEND_NAME = "kccl"

# Collective timeout for the weight-sync groups; BKCL's own default is 600 s.
_DEFAULT_TIMEOUT = timedelta(seconds=int(os.environ.get("BKCL_TIMEOUT", 1800)))

_XCCL_ENABLED = os.environ.get("XCCL_WEIGHT_SYNC", "1") != "0"
_FORCE_FLAT = os.environ.get("XCCL_FORCE_FLAT_PG", "0") == "1"


class PyKunlunCommunicator:
    """Drop-in replacement for ``PyNcclCommunicator`` on KunLun XPU."""

    def __init__(
        self, group: "StatelessProcessGroup", device: Union[int, str, torch.device]
    ):
        self.group = group
        self.rank = group.rank
        self.world_size = group.world_size

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.use_xccl = _XCCL_ENABLED
        self.pg = None  # flat group (single host, or forced flat)
        self._inter = None  # one leader per host
        self._intra = None  # the ranks of this host
        self._backends = []
        self._ready = False

        # world_size == 1 => nothing to communicate with.
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        atexit.register(self._atexit_shutdown)

        # NOTE: nothing is built here on purpose -- see _ensure_ready().

    def _build_pg(self, name: str, ranks: list):
        """Assemble a stateless ProcessGroupXCCL over ``ranks`` (global ids).

        A PrefixStore per group keeps the rendezvous keys of the different
        sub-groups (and of the store's own metadata traffic) apart.
        """
        from torch.distributed import ProcessGroupXCCL

        n = len(ranks)
        local_rank = ranks.index(self.rank)
        store = PrefixStore(name, self.group.store)

        pg = ProcessGroup(store, local_rank, n)
        options = ProcessGroupXCCL.Options()
        if hasattr(options, "_timeout"):
            options._timeout = _DEFAULT_TIMEOUT
        backend = ProcessGroupXCCL(store, local_rank, n, options)
        backend._set_sequence_number_for_group()

        backend_type = ProcessGroup.BackendType.CUSTOM
        pg._set_default_backend(backend_type)
        pg._register_backend(self.device, backend_type, backend)

        self._backends.append(backend)
        return pg

    def _ensure_ready(self):
        """Build the XCCL group(s) on first use, not in ``__init__``."""
        if self._ready:
            return
        if not self.use_xccl:
            self._ready = True
            return

        hosts = self.group.all_gather_obj(socket.gethostname())
        order = []
        for h in hosts:
            if h not in order:
                order.append(h)
        self._hosts = hosts
        self._order = order
        self._members = {
            h: [r for r in range(self.world_size) if hosts[r] == h] for h in order
        }
        self._leaders = [self._members[h][0] for h in order]
        self._my_host = hosts[self.rank]
        self._my_members = self._members[self._my_host]

        if len(order) == 1 or _FORCE_FLAT:
            self.pg = self._build_pg("kunlun_weight_sync", list(range(self.world_size)))
        else:
            # Same construction order on every rank: inter, then intra.
            if self.rank in self._leaders:
                self._inter = self._build_pg("kunlun_ws_inter", self._leaders)
            if len(self._my_members) > 1:
                self._intra = self._build_pg(
                    f"kunlun_ws_intra_{order.index(self._my_host)}", self._my_members
                )
        self._ready = True

    def _prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        """XCCL collectives need a contiguous tensor on this device."""
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        """Broadcast ``tensor`` from rank ``src`` to every rank in the group."""
        if self.disabled:
            return
        self._ensure_ready()
        if not self.use_xccl:
            self._tcp_broadcast(tensor, src)
            return

        work = self._prepare(tensor)
        if self.pg is not None:
            self.pg.broadcast(work, src).wait()
        else:
            src_leader = self._members[self._hosts[src]][0]
            if (
                src != src_leader
                and self._my_host == self._hosts[src]
                and self._intra is not None
            ):
                self._intra.broadcast(work, self._my_members.index(src)).wait()
            if self._inter is not None:
                self._inter.broadcast(work, self._leaders.index(src_leader)).wait()
            if self._intra is not None:
                self._intra.broadcast(work, 0).wait()
        if work is not tensor:
            tensor.copy_(work)

    def _tcp_broadcast(self, tensor: torch.Tensor, src: int):
        """The original pickle-through-the-store path, kept for XCCL_WEIGHT_SYNC=0."""
        if self.rank == src:
            buf = io.BytesIO()
            torch.save(tensor.detach().cpu(), buf)
            data = buf.getvalue()
            for dst in range(self.world_size):
                if dst != src:
                    self.group.send_obj(data, dst=dst)
        else:
            data = self.group.recv_obj(src=src)
            tensor.copy_(
                torch.load(io.BytesIO(data), weights_only=False).to(self.device)
            )

    @staticmethod
    def _allreduce(pg, tensor: torch.Tensor, op):
        if op is None:
            pg.allreduce([tensor]).wait()
            return
        from torch._C._distributed_c10d import AllreduceOptions

        opts = AllreduceOptions()
        opts.reduceOp = op
        pg.allreduce([tensor], opts).wait()

    def all_reduce(self, tensor: torch.Tensor, op=None, stream=None):
        """Real reduction (the original implementation was ``return tensor``)."""
        if self.disabled:
            return tensor
        self._ensure_ready()
        if not self.use_xccl:
            raise RuntimeError("all_reduce requires XCCL; unset XCCL_WEIGHT_SYNC=0")

        want_avg = False
        if op is not None:
            try:
                from torch.distributed import ReduceOp

                if op == ReduceOp.AVG:
                    op, want_avg = ReduceOp.SUM, True
            except Exception:
                pass

        work = self._prepare(tensor)
        if self.pg is not None:
            self._allreduce(self.pg, work, op)
        else:
            if self._intra is not None:
                self._allreduce(self._intra, work, op)
            if self._inter is not None:
                self._allreduce(self._inter, work, op)
            if self._intra is not None:
                self._intra.broadcast(work, 0).wait()
        if want_avg:
            work.div_(self.world_size)
        if work is not tensor:
            tensor.copy_(work)
        return tensor

    def _pair_pg(self, peer: int):
        """The sub-group containing both this rank and ``peer``, if any."""
        if self.pg is not None:
            return self.pg, peer
        if self._intra is not None and peer in self._my_members:
            return self._intra, self._my_members.index(peer)
        if self._inter is not None and peer in self._leaders:
            return self._inter, self._leaders.index(peer)
        return None, None

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        """Send ``tensor`` to rank ``dst``."""
        if self.disabled:
            return
        self._ensure_ready()
        pg, peer = self._pair_pg(dst) if self.use_xccl else (None, None)
        if pg is None:
            # No symmetric XCCL group spans this pair in an asymmetric
            # multi-node layout. Only broadcast is on the hot path, so falling
            # back to the store here costs nothing in practice.
            buf = io.BytesIO()
            torch.save(tensor.detach().cpu(), buf)
            self.group.send_obj(buf.getvalue(), dst=dst)
            return
        pg.send([self._prepare(tensor)], peer, 0).wait()

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        """Receive into ``tensor`` from rank ``src``."""
        if self.disabled:
            return
        self._ensure_ready()
        pg, peer = self._pair_pg(src) if self.use_xccl else (None, None)
        if pg is None:
            data = self.group.recv_obj(src=src)
            tensor.copy_(
                torch.load(io.BytesIO(data), weights_only=False).to(self.device)
            )
            return
        if tensor.device == self.device and tensor.is_contiguous():
            pg.recv([tensor], peer, 0).wait()
            return
        buf = torch.empty_like(tensor, device=self.device)
        pg.recv([buf], peer, 0).wait()
        tensor.copy_(buf)

    def destroy(self):
        """Shut down every XCCL backend and mark the communicator unusable."""
        for backend in getattr(self, "_backends", []):
            try:
                backend.shutdown()
            except Exception:
                pass
        self._backends = []
        self.pg = None
        self._inter = None
        self._intra = None
        self.available = False
        self.disabled = True

    def _atexit_shutdown(self):
        """atexit hook: tear XCCL down before the store goes away."""
        try:
            self.destroy()
        except Exception:
            pass
