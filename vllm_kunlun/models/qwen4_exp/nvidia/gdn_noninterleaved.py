"""Q38 XPU PORT: 昆仑 GDN 的"非交错"权重布局 + sigmoid 输出门适配。

背景一：权重布局
----------------
vllm_kunlun 的 Qwen3NextGatedDeltaNet 只实现了 Qwen3-Next 的交错布局：
in_proj_qkvz 是**一个** fused shard（output_sizes=[k+k+v+v]），
fix_query_key_value_ordering 按 [ng, (hk+hk+np/ng*hv+np/ng*hv)] 解交错。

而 Qwen3.8-Flash-Next 的 ckpt 与 Qwen3.5 一致，q/k/v/z 是**分开存**的：
linear_attn.in_proj_qkv + linear_attn.in_proj_z（in_proj_b /
in_proj_a 同理）。上游 Qwen3_5Model.hf_to_vllm_mapper 用
orig_to_new_stacked 把它们映射成 in_proj_qkvz 的 shard (0,1,2) 与 3，
因此运行时的 in_proj_qkvz 必须是 **4 个** output shard。直接用昆仑类会在
加载权重时报 ValueError: Shard id index 1 should be between 0 and 0。

这里继承昆仑实现，只覆盖两个与布局相关的方法：投影切分改成 4 shard，
qkvz/ba 拆分改成沿最后一维直接切（等价于上游
QwenGatedDeltaNetAttention 在 gqa_interleaved_layout=False 分支的行为），
核心计算仍走昆仑 XPU 算子。

背景二：输出门激活函数
----------------------
上游 QwenGatedDeltaNetAttention 是这样建输出 norm 的::

    output_gate_type = getattr(config, "output_gate_type", "silu")
    self.norm = RMSNormGated(..., activation=output_gate_type)

而 vllm_kunlun 用的是自带的 vllm_kunlun.ops.fla.RMSNormGated，它**没有
activation 参数**，写死 ``norm(x) * silu(z)``。Qwen3.8-Flash-Next 的
config 里 ``output_gate_type = "sigmoid"``，于是 36 个 linear_attention
层全部多乘了一个 z（silu(z) = z * sigmoid(z)）：数值量级看着正常、不
NaN、不爆，但语义全错，表现为 logits 接近均匀分布、生成结构化乱码。

这里在 ``super().__init__()`` 之后按 config 把 self.norm 换成
sigmoid 门实现（参数名仍为 ``norm.weight``，权重加载不受影响），
等价于上游 ``RMSNormGated(activation="sigmoid", norm_before_gate=True)``
的 forward_static。
"""

import torch
from torch import nn

import os as _q38_os

try:
    import kunlun_ops as _q38_kunlun_ops
except ImportError:
    _q38_kunlun_ops = None

# Q38 FUSE: 置 Q38_GDN_FUSED=0 回退原始逐算子实现
_Q38_GDN_FUSED = (
    _q38_os.environ.get("Q38_GDN_FUSED", "1") == "1" and _q38_kunlun_ops is not None
)

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm_kunlun.models.qwen3_next import Qwen3NextGatedDeltaNet


class SigmoidGatedRMSNorm(nn.Module):
    """``norm(x) * sigmoid(z)``，对应 config.output_gate_type == "sigmoid"。

    与上游 RMSNormGated(group_size=None, norm_before_gate=True,
    activation="sigmoid").forward_static 等价：fp32 中间计算，再转回原 dtype。
    """

    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.group_size = None
        self.norm_before_gate = True
        self.activation = "sigmoid"
        self.weight = nn.Parameter(
            torch.empty(hidden_size, device=device, dtype=dtype)
        )
        self.register_parameter("bias", None)
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        # Q38 FUSE: kunlun_ops.rmsnorm 与 fp32 torch 参考逐位一致（UT rel=0.0）；
        # sigmoid 门 bf16 计算，整体 rel~3e-3（bf16 1ulp 舍入）。10 个
        # elementwise kernel -> 3 个（实测 0.143ms -> 0.036ms）。
        if _Q38_GDN_FUSED and z is not None and x.dim() == 2 and x.is_contiguous():
            out = torch.empty_like(x)
            _q38_kunlun_ops.rmsnorm(x, self.weight, out, self.eps)
            return out * torch.sigmoid(z)
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps) * self.weight.float()
        if z is not None:
            out = out * torch.sigmoid(z.float())
        return out.to(orig_dtype)


class KunlunGDNNonInterleaved(Qwen3NextGatedDeltaNet):
    """昆仑 GDN + Qwen3.5/Qwen3.8 的 q/k/v/z 分离权重布局 + sigmoid 输出门。"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        gate_type = getattr(config, "output_gate_type", "silu")
        if gate_type == "sigmoid":
            self.norm = SigmoidGatedRMSNorm(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                device=self.norm.weight.device,
                dtype=self.norm.weight.dtype,
            )

    def create_qkvz_proj(
        self, hidden_size, key_dim, value_dim, quant_config, prefix
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        n = mixed_qkvz.shape[0]
        kd = self.key_dim // self.tp_size
        vd = self.value_dim // self.tp_size
        query, key, value, z = mixed_qkvz.split([kd, kd, vd, vd], dim=-1)
        query = query.reshape(n, -1, self.head_k_dim)
        key = key.reshape(n, -1, self.head_k_dim)
        value = value.reshape(n, -1, self.head_v_dim)
        z = z.reshape(n, -1, self.head_v_dim)
        b, a = mixed_ba.chunk(2, dim=-1)
        b = b.reshape(n, self.num_v_heads // self.tp_size)
        a = a.reshape(n, self.num_v_heads // self.tp_size)
        return query, key, value, z, b, a


    def forward(self, hidden_states, output):
        # Q38 FUSE: q/k/v 本就是 in_proj_qkvz 输出的前三个连续列块，一次
        # contiguous 切片替代基类的 3 次 reshape 拷贝 + 1 次 cat。
        if not _Q38_GDN_FUSED:
            return super().forward(hidden_states, output)
        num_tokens = hidden_states.size(0)
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        kd = self.key_dim // self.tp_size
        vd = self.value_dim // self.tp_size
        qkv_dim = 2 * kd + vd
        mixed_qkv = projected_states_qkvz[:, :qkv_dim].contiguous()
        z = projected_states_qkvz[:, qkv_dim:]
        b, a = projected_states_ba.chunk(2, dim=-1)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.gdn_attention_core(
            mixed_qkv, b, a, core_attn_out, self.prefix
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(num_tokens, -1)
        output[:num_tokens], _ = self.out_proj(core_attn_out)
