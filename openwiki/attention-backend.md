---
type: reference
title: Attention Backend 与 KV Cache
summary: >-
  四个注册 backend（KunlunAttention / FlashMLA / FlashMLASparse / GDN）加一条
  非 paged 编码器路径，各自的 KV cache 布局、图捕获限制、前缀缓存与 chunked prefill 约束。
generated:
  by: hand-authored (Claude Code, OpenWiki OKF v0.2 conventions)
  at: 2026-09-02T00:00:00Z
evidence_version:
  repo: https://github.com/baidu/vLLM-Kunlun
  ref: v0.25.1-dev
  commit: c53e090ff8800f586bf9e36e0d876779981bfb20
sources:
- repo://vllm_kunlun/v1/attention/backends/kunlun_attn.py#L62-L1013
- repo://vllm_kunlun/v1/attention/backends/mla/flashmla.py#L30-L138
- repo://vllm_kunlun/v1/attention/backends/mla/flashmla_sparse.py#L63-L182
- repo://vllm_kunlun/v1/attention/backends/mla/common.py#L280-L1969
- repo://vllm_kunlun/v1/attention/backends/gdn_attn.py#L25-L512
- repo://vllm_kunlun/ops/paged_attn.py#L12-L55
- repo://vllm_kunlun/ops/attention/layer.py#L129-L170
- repo://vllm_kunlun/ops/attention/flashmla.py#L34-L259
- repo://vllm_kunlun/v1/worker/block_table.py#L3-L47
claims: .claims/attention-backend.json
---

# Attention Backend 与 KV Cache

选择逻辑在 [platform-contract.md](platform-contract.md#4-get_attn_backend_cls忽略用户选择)：
`get_attn_backend_cls` **忽略** `selected_backend`，只按模型特征分派。

## 1. 五条路径

| Backend | `get_name()` | 何时使用 | 位置 |
| --- | --- | --- | --- |
| `KunlunAttentionBackend` | `"CUSTOM"` | 默认（所有非 MLA 模型） | `kunlun_attn.py#L62-L82` |
| `FlashMLABackend` | `"FLASHMLA"` | MLA 模型（DeepSeek V3 等） | `mla/flashmla.py#L30-L46` |
| `FlashMLASparseBackend` | `"FLASHMLA_SPARSE_VLLM_V1"` | MLA + `index_topk`（DeepSeek V3.2 DSA） | `mla/flashmla_sparse.py#L63-L69` |
| `GDNAttentionBackend` | `"GDN_ATTN"`，`is_ssm() → True` | 由模型声明 `mamba_type()` 选中 | `gdn_attn.py#L25-L36` |
| 非 paged 编码器路径 | 强制 `_Backend.FLASH_ATTN` | encoder / 无 KV cache 场景 | `ops/attention/layer.py#L129-L170` |

第五条路径不经过 backend 注册表，直接调 `flash_attn_func`。

GDN 的选中方式与其他四个不同——它由**模型**声明：
`models/qwen3_next.py#L264-L265` 返回 `MambaAttentionBackendEnum.GDN_ATTN`。
并且 `gdn_attn.py#L505-L512` 在 import 时**就地替换上游符号**
（属于 [post-import 补丁](architecture.md#42-post-import-就地补丁8-个)）。

## 2. KV Cache 布局

| Backend | shape | dtype | 备注 |
| --- | --- | --- | --- |
| `KunlunAttention` | `(2, num_blocks, num_kv_heads, block_size, head_size)`（BHLD） | fp16 / bf16 | `ops/paged_attn.py#L42-L55` |
| `FlashMLA` | `(num_blocks, block_size, head_size)`，`head_size = 576` | fp16 / bf16 | `mla/common.py#L280-L324` |
| `FlashMLASparse` | 656 字节/token 的 fp8 布局，或 bf16 | fp8 / bf16 | `flashmla_sparse.py#L83-L104` |
| `GDN` | `MambaSpec`（SSM state，非 paged KV） | — | `gdn_attn.py#L98` |

其他常量：

- 支持的 head size：`[32, 64, 80, 96, 112, 120, 128, 192, 256, 512]`（`ops/paged_attn.py#L36-L40`）
- `_PARTITION_SIZE = 512`（`ops/paged_attn.py#L12`）
- `reshape_and_cache_flash(..., BLHD_LAYOUT=False)`（`kunlun_attn.py#L820-L827`）
- sparse MLA 默认 `topk_tokens = 2048`、`block_size = 64`（`flashmla_sparse.py#L152-L182`）
- GDN 只接受 `--mamba-cache-mode=align`（`models/qwen3_next.py#L1373-L1376`）

> ⚠️ **fp8 的矛盾**：`check_if_supports_dtype` 只允许
> `{fp32, fp16, bf16, int8}`（`platforms/kunlun.py#L390-L405`），
> 但 sparse MLA 的 KV cache 有 fp8 布局。两者不冲突——前者约束的是
> 模型权重/激活 dtype，KV cache dtype 由 `--kv-cache-dtype` 单独控制
> （tutorial 里 DeepSeek-V3.2 用 `--dtype float16 --kv-cache-dtype bfloat16`）。

### 混合 KV cache 的 block table 缩放

`support_hybrid_kv_cache = True`（`platforms/kunlun.py#L413-L415`）。
混合模型（如 Qwen3-Next：线性注意力层 + 标准注意力层交错）下，
不同层的 page 大小不同，运行时需要缩放 block table：

- `kunlun_attn.py#L692-L713`、`#L854-L855` —— `_get_block_table_scale`
- `v1/worker/mamba_utils.py#L315-L340` —— 对应的 stride 改写

两处必须一致，改一处必须改另一处。

### slot mapping 换成原生算子

`v1/worker/block_table.py#L3-L47` 把 slot mapping 计算换成
`kunlun_ops.compute_slot_mappings`。**原因**：`torch.searchsorted`
在 XPU 上会**静默回落到 CPU**，成为逐 step 的同步点。
这是个典型的"能跑但慢得离谱"类问题，值得记住。

## 3. 图捕获与 cascade attention

- MLA / GDN 的图捕获**只覆盖 decode**（prefill 走 eager）。
- **cascade attention 无条件不支持**：`kunlun_attn.py#L992-L1013`
  直接 `return False`。
- `KVBlockZeroer._zero_block_ids` 被改成 `return` 空实现
  （`v1/worker/utils.py#L110-L112`）——block 复用时不清零。

## 4. MLA 细节

### chunked prefill workspace

`mla/common.py#L466-L495`：workspace 大小
`max(8 * max_model_len, 4 * max_num_seqs * block_size)`，上限 64K。
这个值决定 chunked prefill 单次能处理多长的 context。

### KV sharing 不支持

`mla/common.py#L1024-L1025`：MLA 路径下 KV sharing 直接
`NotImplementedError`。

### int8 BMM 开关

唯一有真实消费者的自定义环境变量 `VLLM_KUNLUN_ENABLE_INT8_BMM`：

- `mla/common.py#L1152-L1172` —— 开启时走 `kunlun_ops.mla_bmm_I8(...)`，
  否则 `torch.bmm(x, self.W_UV, out=out)`
- `mla/common.py#L1459-L1484` —— int8 权重预处理
  `kunlun_ops.quant2d(w_uk_dq_trans, self.W_UK_T, self.W_UK_SCALE)`
- 第三个引用点 `#L1969`

### prefill helper 与数值一致性风险

`mla/common.py#L1272-L1305` 的 `_flash_attn_varlen_diff_headdims`
把上游 FlashAttention 调用（`#L1272-L1278`，已注释）换成
`kunlun_ops.attention` + **硬编码的 DeepSeek alpha**
`ds_alpha = 1.8738542070926265`（`#L1280`）+ 手工构造的 `-inf` LSE。

配套地，稀疏 prefill 的 LSE 有一处**已记录的 scale 不匹配**：
`ops/attention/flashmla.py#L256-L259`，关系式
`gpu_lse * out_scale = kunlun_lse`，其中 `out_scale = 1 / math.log2(math.e)`。

**这两处是做数值对齐（numerical parity）排查时最应该先看的地方。**

### 两个名字骗人的辅助函数

| 函数 | 名字暗示 | 实际行为 | 位置 |
| --- | --- | --- | --- |
| `is_flashmla_supported()` | 做能力探测 | **永远** `return True, None` | `ops/attention/flashmla.py#L34-L38` |
| `get_mla_metadata()` | 返回 tile scheduler metadata | 返回 `cache_seqlens_cpu, cache_seqlens` | `ops/attention/flashmla.py#L56-L58` |

与第二点对应：FlashMLA builder 里 CUDA-graph 持久 buffer 那段被整块注释掉
（`mla/flashmla.py#L113-L138`），但 `(num_sms, 8)` 的 buffer 仍然在分配
（`#L81-L96`，注释 `# TileSchedulerMetaDataSize = 8`）。

## 5. 前缀缓存与 chunked prefill

仓库里有一个独立的硬件诊断脚本
`vllm_kunlun/tests/test_prefill_attention_prefix_cache.py`。
它**不是 pytest**——通过退出码表达结果，用于定位前缀缓存下
prefill attention 的正确性问题。参见 [testing-and-ci.md](testing-and-ci.md)。

tutorial 里 DeepSeek-V3.2 的启动命令显式关掉了两者：
`--no-enable-chunked-prefill --no-enable-prefix-caching`。
这是官方给出的保守配置，见 [model-support.md](model-support.md)。

## 6. 投机解码相关的 attention 路径

- `kunlun_attn.py#L663` —— `is_speculative = self.reorder_batch_threshold > 1`
- `kunlun_attn.py#L962-L987` —— spec decode attention：
  `assert query_seq_len % batch_size == 0`，`qlen = query_seq_len // batch_size`，
  `scale=0.0`
- `kunlun_attn.py#L913-L948` —— `speculative_attention`，用
  `inspect.signature(...).parameters` 运行时探测厂商 kernel 是否支持
  `max_window_size`；`block_size = key_cache.shape[2]`

完整投机解码链路见 [spec-decode-and-sampling.md](spec-decode-and-sampling.md)。

## 相关页面

- [platform-contract.md](platform-contract.md) —— backend 选择与 block size 强制
- [linear-attention.md](linear-attention.md) —— GDN / FLA / Mamba 路径
- [spec-decode-and-sampling.md](spec-decode-and-sampling.md)
- [known-gaps.md](known-gaps.md) —— 名字骗人的函数与死代码汇总
