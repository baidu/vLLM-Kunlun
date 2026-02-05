import torch
from typing import Any, Literal, Optional, cast, Callable, Optional

from compressed_tensors.config import (CompressionFormat,
                                       SparsityCompressionConfig,
                                       SparsityStructure)
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationStrategy)
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.quantization.utils import replace_parameter
# TODO: import position will be changed after 0.9.0
# vllm.model_executor.layers.fused_moe.fused_moe --> vllm.model_executor.layers.fused_moe

from vllm.model_executor.utils import set_weight_attrs
import re
import kunlun_ops


from safetensors.torch import load_file as safe_load_file

class CompressedTensorsMoEMethod(FusedMoEMethodBase):

    def get_moe_method(quant_config, layer) -> "CompressedTensorsMoEMethod":
        tsm = getattr(quant_config, "target_scheme_map", None) or {}
        linear_cfg = None
        for k in ("Linear", "FusedMoE", "MoE", "Moe", "Experts"):
            if k in tsm and isinstance(tsm[k], dict):
                linear_cfg = tsm[k]; break
        if not linear_cfg:
            # print("target_scheme_map missing; fallback to INT8(W8A8) method")
            return CompressedTensorsW8A8Int8MoEMethod(quant_config)
        wq = linear_cfg.get("weights"); aq = linear_cfg.get("input_activations")
        if not wq or not aq:
            # print("incomplete scheme; fallback to INT8(W8A8)")
            return CompressedTensorsW8A8Int8MoEMethod(quant_config)
        # 其它分流按需；默认回落：
        return CompressedTensorsW8A8Int8MoEMethod(quant_config)

# copied from vllm 0.9.0
class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        
        # 直接创建默认的量化配置字典，避免 QuantizationArgs 的验证问题
        # print("Creating default INT8 quantization config for MoE")
        
        # 创建默认的权重量化配置字典
        self.weight_quant = type('WeightQuant', (), {
            'type': 'int',
            'num_bits': 8,
            'strategy': 'channel',
            'group_size': 128,
            'symmetric': True,
            'dynamic': False,
            'actorder': 'none',
            'observer': None,
            'observer_kwargs': {},
            'block_structure': None
        })()
        
        # 创建默认的输入激活量化配置字典
        self.input_quant = type('InputQuant', (), {
            'type': 'int',
            'num_bits': 8,
            'strategy': 'token',
            'group_size': 128,
            'symmetric': True,
            'dynamic': True,
            'actorder': 'none',
            'observer': None,
            'observer_kwargs': {},
            'block_structure': None
        })()

        # 修改比较方式，直接比较字符串
        per_channel = (
            self.weight_quant.strategy == "channel"
            and self.input_quant.strategy == "token")
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")

    def create_weights1(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        # 权重先用浮点占位，便于从 ckpt 加载原始权重
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),  # 通常是 torch.bfloat16
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # 通道 scale：float32 + 二维 [E, out]（与 fused_moe/UT 对齐）
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32),
            requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # 输入 scale 动态计算即可
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=torch.int8),  # 直接使用 int8
            requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=torch.int8),  # 直接使用 int8
            requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # 缩放因子
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32),
            requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # 输入 scale 动态计算
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        
    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return
        #原始权重转 float32 做统计更稳健
        w13_f = layer.w13_weight.float()
        w2_f  = layer.w2_weight.float()

        # 每列(abs_max) -> per-column scale（out 维在 dim=1，列在 dim=-1）
        qmax = 127.0
        w13_abs_max = torch.amax(torch.abs(w13_f), dim=-1)  # [E, 2N]
        w2_abs_max  = torch.amax(torch.abs(w2_f),  dim=-1)  # [E, H]

        w13_scale_2d = torch.clamp(w13_abs_max, min=1e-6) / qmax  # [E, 2N], float32
        w2_scale_2d  = torch.clamp(w2_abs_max,  min=1e-6) / qmax  # [E, H],  float32

        # 量化：用 3D scale 广播，存回 2D scale
        w13_scale_3d = w13_scale_2d.unsqueeze(-1)  # [E, 2N, 1]
        w2_scale_3d  = w2_scale_2d.unsqueeze(-1)   # [E, H, 1]

        w13_q = torch.round(w13_f / w13_scale_3d).clamp_(-128, 127).to(torch.int8)
        w2_q  = torch.round(w2_f  / w2_scale_3d ).clamp_(-128, 127).to(torch.int8)

        # 可选：若你的 fused/kernel 期望 scale 预乘 127（与某些 UT 后端一致），打开下面两行：
        w13_scale_2d = w13_scale_2d * 127.0
        w2_scale_2d  = w2_scale_2d  * 127.0

        # 回写参数：权重 int8；scale 用 float32 + 2D
        replace_parameter(layer, 'w13_weight', torch.nn.Parameter(w13_q, requires_grad=False))
        replace_parameter(layer, 'w2_weight',  torch.nn.Parameter(w2_q,  requires_grad=False))
        replace_parameter(layer, 'w13_weight_scale',
                        torch.nn.Parameter(w13_scale_2d.contiguous(), requires_grad=False))
        replace_parameter(layer, 'w2_weight_scale',
                        torch.nn.Parameter(w2_scale_2d.contiguous(),  requires_grad=False))

        # 简要检查
        print(f"w13: {w13_q.shape}, w13_s: {w13_scale_2d.shape}, w2: {w2_q.shape}, w2_s: {w2_scale_2d.shape}")
 
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,  # 添加这个参数
        expert_load_view: Optional[torch.Tensor] = None,  # 添加这个参数
        logical_to_physical_map: Optional[torch.Tensor] = None,  # 添加这个参数
        logical_replica_count: Optional[torch.Tensor] = None,  # 添加这个参数
        linear_weights: Optional[torch.Tensor] = None,  # 添加这个参数
    ) -> torch.Tensor:

        output = torch.empty_like(x)
        torch.ops._C.moe_ffn_per_token_block(
            x=x,
            inter_weight=layer.w13_weight,
            inter_scale=layer.w13_weight_scale,
            outer_weight=layer.w2_weight,
            outer_scale=layer.w2_weight_scale,
            top_k=top_k,
            global_num_experts=global_num_experts,
            linear_weights=linear_weights,
            expert_map=expert_map,
            activation=activation,
            output=output,
            use_expert_parallel=expert_map is not None,
            ep_size=expert_map.size(0) if expert_map is not None else 1,
            ep_rank=0,
        )
        return output

print("[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsMoEMethod \
      --> vllm_xpu.model_executor.layers.quantization.compressed_tensors_moe.py:CompressedTensorsMoEMethod")