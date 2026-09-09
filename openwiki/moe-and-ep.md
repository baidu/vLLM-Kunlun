---
type: reference
title: Fused MoE 与专家并行
summary: >-
  UnquantizedFusedMoEMethod 的 OOT 实现：M*top_k > 768 的大小 batch 分支、
  act=None + 显式 silu_and_mul 这个规避 kernel bug 的必要写法，以及 EP 路径的 Python 逐专家循环。
generated:
  by: hand-authored (Claude Code, OpenWiki OKF v0.2 conventions)
  at: 2026-09-02T00:00:00Z
evidence_version:
  repo: https://github.com/baidu/vLLM-Kunlun
  ref: v0.25.1-dev
  commit: c53e090ff8800f586bf9e36e0d876779981bfb20
sources:
- repo://vllm_kunlun/ops/fused_moe/layer.py#L16-L89
- repo://vllm_kunlun/ops/_kunlun_ops.py#L377-L680
- repo://vllm_kunlun/quantization/compressed_tensors/compressed_tensors_moe.py#L149-L321
- repo://vllm_kunlun/platforms/kunlun.py#L257-L274
- repo://setup_env.sh
claims: .claims/moe-and-ep.json
---

# Fused MoE 与专家并行

## 1. 挂载点

`ops/fused_moe/layer.py#L16-L17` ——
`@CustomOp.register_oot(name="UnquantizedFusedMoEMethod")`，
属于 [OOT 层注册](architecture.md#43-oot-层注册)。

`apply_monolithic`（`#L43-L89`）按 `self.moe.use_ep` 二分：

```mermaid
graph TD
    A["apply_monolithic"] --> B{"self.moe.use_ep"}
    B -- False --> C["fused_moe<br/>_kunlun_ops.py#L377-L621"]
    B -- True --> D["fused_moe_ep<br/>_kunlun_ops.py#L623-L680"]
    C --> C1["routing:<br/>moe_softmax_topk_norm(stable=True)<br/>或 moe_sigmoid_group_topk_norm"]
    C1 --> C2{"M * top_k > 768?"}
    C2 -- 是 --> C3["gen_block_statistic<br/>+ moe_pre_sorted"]
    C2 -- 否 --> C4["moe_pre_small"]
    C3 --> C5["moe_fc(act=None)"]
    C4 --> C5
    C5 --> C6["显式 silu_and_mul"]
    C6 --> C7["moe_fc"]
    C7 --> C8["moe_post"]
    D --> D1["moe_softmax_topk"]
    D1 --> D2["Python 逐专家 for 循环<br/>#L661-L673"]
```

## 2. 非 EP 路径（`fused_moe`）

`ops/_kunlun_ops.py#L377-L621`。

**路由**：`moe_softmax_topk_norm(stable=True)` 或
`moe_sigmoid_group_topk_norm`（后者用于 DeepSeek 系的分组 topk）。

**大小 batch 分支**（`#L528`）：

```python
if M * moe_top_k > 768:
    # gen_block_statistic + moe_pre_sorted
else:
    # torch.ops.xspeedgate_ops.moe_pre_small
```

`M` 是 token 数。也就是说同一个 MoE 层在 prefill 和 decode 下走的是
**两套不同 kernel**，切换点在 `M * top_k = 768`。

`#L523-L526` 通过 `current_workspace_manager().get_simultaneous(...)` 拿工作区。
具体算子调用在 `#L547-L619`：`torch.ops._C.gen_block_statistic` /
`moe_pre_sorted` / `moe_fc` / `silu_and_mul` / `moe_post`。

`#L478-L496` 还有一条 `M * moe_top_k < 400` 的更小 batch 路径，**整块被注释掉**。

## 3. 关键：`act=None` + 显式 `silu_and_mul` 不是冗余写法

`ops/_kunlun_ops.py#L505-L519` 有一段必须原文引用的 NOTE：

> 融合的 `moe_fc(act="SWISH_GLU")` 对**所有 `M >= 1024`** 数值都是错的，
> 并且它"**was the source of the multi-concurrent 'garbled output' symptom**"。

所以代码**永远**用 `moe_fc(act=None)` 拿到中间结果，再显式调一次
`silu_and_mul`，然后第二个 `moe_fc`。

**这是本仓库最重要的一条"不要优化掉"的注释。**任何人看到这里想"两次
kernel launch 可以融成一次"，都会重新引入高并发下输出乱码的 bug。
排查"并发一高就输出乱码"类问题时，也应先确认这条规避是否还在。

## 4. 量化 int8 单体路径

`quantization/compressed_tensors/compressed_tensors_moe.py#L149-L321`
是上面 `fused_moe` 的 int8 镜像，**同样的 768 阈值**在 `#L200`。
两份实现需要同步维护。

## 5. EP 路径是 Python 逐专家循环

`ops/_kunlun_ops.py#L623-L680` 的 `fused_moe_ep`：
`torch.ops._C.moe_softmax_topk` 做路由，然后 `#L661-L673`
是一个**逐专家的 Python `for` 循环**。

**性能含义**：非 EP 路径是单个融合 kernel 处理所有专家，EP 路径是
`num_local_experts` 次独立 kernel launch。两条路径存在**量级上的性能不对称**。

这也解释了两件事：

1. 特性矩阵（`supported_features.md#L8-L14`）声称专家并行 🟢，
   但**所有 8 篇 tutorial 都是 TP-only**，没有一条命令用
   `--enable-expert-parallel` 或 `--data-parallel-size`。
2. `check_and_update_config` 里对 DeepEP 高吞吐 + DP>1 的组合直接
   **强制 full eager**（`platforms/kunlun.py#L257-L274`）——
   这条路径显然还没有和图捕获一起验证过。

## 6. 相关环境变量

- `ENABLE_VLLM_MOE_FC_SORTED`（`platforms/envs.py#L53`，注释
  `fuse sorted op with fused_moe kernel`）——**没有消费者**，见
  [architecture.md](architecture.md#7-环境变量两套互不相交的集合)。
- `XPU_USE_MOE_SORTED_THRES` —— 厂商 legacy `moe_ffn_block` 根据 token
  行数 `M` 选择 sorted 路径的阈值，默认 `512`。当前 vLLM-Kunlun 的
  `fused_moe` 路径使用内部条件，不读取这个变量，因此启动脚本不再设置它。

## 相关页面

- [quantization.md](quantization.md) —— AWQ MoE 回落上游、W4A8 未实现
- [architecture.md](architecture.md) —— OOT 层注册机制
- [model-support.md](model-support.md) —— MoE 模型清单
