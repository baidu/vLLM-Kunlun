# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


@PluggableLayer.register_oot(name="ReplicatedLinear")
class KunlunReplicatedLinear(ReplicatedLinear):
    """Kunlun OOT replacement for vLLM's ReplicatedLinear."""

    def get_weights(self):
        if hasattr(self, "kunlun_linear_weights"):
            return self.kunlun_linear_weights
        weights = torch.nn.Parameter(self.weight.to(torch.float32))
        self.register_parameter("kunlun_linear_weights", weights)
        return self.kunlun_linear_weights

    def get_weights_half(self):
        if hasattr(self, "kunlun_linear_weights_half"):
            return self.kunlun_linear_weights_half
        weights = torch.nn.Parameter(self.weight.to(torch.float16))
        self.register_parameter("kunlun_linear_weights_half", weights)
        return self.kunlun_linear_weights_half


@PluggableLayer.register_oot(name="MergedColumnParallelLinear")
class KunlunMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Kunlun OOT replacement for vLLM's MergedColumnParallelLinear."""

    @staticmethod
    def _adjust_bitblas_shard(param, shard_size, shard_offset):
        tile_size = getattr(param, "bitblas_tile_size", None)
        if tile_size is not None:
            return shard_size // tile_size, shard_offset // tile_size
        return shard_size, shard_offset

    @staticmethod
    def _adjust_marlin_shard(param, shard_size, shard_offset):
        tile_size = getattr(param, "marlin_tile_size", None)
        if tile_size is None:
            return shard_size, shard_offset
        return shard_size * tile_size, shard_offset * tile_size

    @staticmethod
    def _adjust_block_scale_shard(weight_block_size, shard_size, shard_offset):
        assert weight_block_size is not None
        block_n = weight_block_size[0]
        shard_offset = (shard_offset + block_n - 1) // block_n
        shard_size = (shard_size + block_n - 1) // block_n
        return shard_size, shard_offset

    @staticmethod
    def _adjust_bitsandbytes_shard(param, shard_offsets, shard_id):
        total, _ = shard_offsets["total"]
        orig_offset, orig_size = shard_offsets[shard_id]
        quantized_total = param.data.shape[0]
        return (
            orig_size * quantized_total // total,
            orig_offset * quantized_total // total,
        )

    @staticmethod
    def _adjust_scalar_to_array(param, loaded_weight, shard_id):
        shard_id = {"q": 0, "k": 1, "v": 2}.get(shard_id, shard_id)
        if not isinstance(shard_id, int):
            raise ValueError(f"Unknown shard id {shard_id}")
        if loaded_weight.ndim != 0:
            assert loaded_weight.shape[0] == 1
            loaded_weight = loaded_weight[0]
        return param[shard_id], loaded_weight

    def _load_single_shard(
        self, param, loaded_weight, output_dim, shard_id, shard_offset, shard_size
    ):
        if isinstance(param, BlockQuantScaleParameter):
            shard_size, shard_offset = self._adjust_block_scale_shard(
                getattr(self, "weight_block_size", None), shard_size, shard_offset
            )

        packed_dim = getattr(param, "packed_dim", None)
        if packed_dim == output_dim:
            shard_size //= param.packed_factor
            shard_offset //= param.packed_factor
            shard_size, shard_offset = self._adjust_marlin_shard(
                param, shard_size, shard_offset
            )
        return self._adjust_bitblas_shard(param, shard_size, shard_offset)

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None
    ):
        is_gguf = getattr(param, "is_gguf_weight", False)
        is_gguf_type = getattr(param, "is_gguf_weight_type", False)
        if isinstance(loaded_shard_id, tuple) and (is_gguf or is_gguf_type):
            raise NotImplementedError("GGUF does not support multiple shard ids.")

        if is_gguf_type:
            if loaded_shard_id is not None:
                param.data[loaded_shard_id].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    i: loaded_weight.item() for i, _ in enumerate(self.output_sizes)
                }
            return

        output_dim = getattr(param, "output_dim", None)
        if is_gguf and loaded_shard_id is not None:
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            loaded_weight = loaded_weight.narrow(
                output_dim, self.tp_rank * shard_size, shard_size
            )
            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

        param_data = param.data
        needs_scalar = getattr(param, "needs_scalar_to_array", False)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if output_dim is None:
                if needs_scalar:
                    param_data, loaded_weight = self._adjust_scalar_to_array(
                        param_data, loaded_weight, 0
                    )
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return

            output_sizes = (
                self.output_sizes[loaded_shard_id[0] : loaded_shard_id[-1] + 1]
                if loaded_shard_id is not None
                else self.output_sizes
            )
            use_bnb = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bnb and isinstance(loaded_shard_id, tuple):
                raise NotImplementedError("BNB does not support multiple shard ids.")

            current_offset = 0
            for shard_id, shard_size in enumerate(output_sizes):
                shard_size, shard_offset = self._load_single_shard(
                    param,
                    loaded_weight,
                    output_dim,
                    shard_id,
                    current_offset,
                    shard_size,
                )
                if use_bnb:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = self._adjust_bitsandbytes_shard(
                        param, offsets, str(shard_id)
                    )
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
                current_offset += output_sizes[shard_id]
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            shard_offset //= self.tp_size
            shard_size //= self.tp_size
            shard_size, shard_offset = self._load_single_shard(
                param,
                loaded_weight,
                output_dim,
                loaded_shard_id,
                shard_offset,
                shard_size,
            )
            use_bnb = getattr(param, "use_bitsandbytes_4bit", False)
            is_sharded = getattr(param, "is_sharded_weight", False) or use_bnb
            if use_bnb:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if not is_sharded:
                loaded_weight = loaded_weight.narrow(
                    output_dim, self.tp_rank * shard_size, shard_size
                )
        elif needs_scalar:
            param_data, loaded_weight = self._adjust_scalar_to_array(
                param_data, loaded_weight, loaded_shard_id
            )
        elif not getattr(param, "ignore_warning", False):
            logger.warning(
                "Loading a weight without output_dim in "
                "MergedColumnParallelLinear; assuming it is replicated."
            )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self, param, loaded_weight, output_sizes=None
    ):
        current_offset = 0
        output_sizes = output_sizes or self.output_sizes
        for shard_id, original_shard_size in enumerate(output_sizes):
            shard_size = original_shard_size
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=current_offset
                )
            else:
                shard_offset = current_offset
            loaded_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_shard, shard_id)
            current_offset += original_shard_size

    def _validate_shard_id(self, loaded_shard_id):
        if loaded_shard_id is None:
            return
        if isinstance(loaded_shard_id, tuple):
            if any(not 0 <= idx < len(self.output_sizes) for idx in loaded_shard_id):
                raise ValueError(f"Invalid shard id {loaded_shard_id}.")
            if len(loaded_shard_id) > 1 and any(
                b - a != 1 for a, b in zip(loaded_shard_id[:-1], loaded_shard_id[1:])
            ):
                raise ValueError(f"Shard ids must be consecutive: {loaded_shard_id}.")
        elif isinstance(loaded_shard_id, int) and not 0 <= loaded_shard_id < len(
            self.output_sizes
        ):
            raise ValueError(f"Invalid shard id {loaded_shard_id}.")

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        self._validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                if isinstance(loaded_shard_id, tuple):
                    for shard_id in loaded_shard_id:
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=shard_id
                        )
                else:
                    param.load_merged_column_weight(
                        loaded_weight=loaded_weight, shard_id=0
                    )
                return
            if type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            if isinstance(param, BlockQuantScaleParameter):
                output_sizes = [
                    self._adjust_block_scale_shard(
                        getattr(self, "weight_block_size", None), size, 0
                    )[0]
                    for size in (output_sizes or self.output_sizes)
                ]
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        if isinstance(param, BlockQuantScaleParameter):
            shard_size, shard_offset = self._adjust_block_scale_shard(
                getattr(self, "weight_block_size", None), shard_size, shard_offset
            )
        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )


def _create_unquantized_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    weight = Parameter(
        torch.empty(
            sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype
        ),
        requires_grad=False,
    )
    set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
    layer.register_parameter("weight", weight)
    set_weight_attrs(weight, extra_weight_attrs)


# This is a quantization-method compatibility patch, not a layer replacement.
UnquantizedLinearMethod.create_weights = _create_unquantized_weights
if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
    WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")
