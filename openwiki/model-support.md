---
type: reference
title: 模型支持
summary: >-
  10 个注册模型各自为什么需要 OOT 实现，DeepSeek V3.2 的检测方式与 MTP，
  LoRA 的现状，以及官方 tutorial 的实际启动命令（全部 TP-only）。
generated:
  by: hand-authored (Claude Code, OpenWiki OKF v0.2 conventions)
  at: 2026-09-02T00:00:00Z
evidence_version:
  repo: https://github.com/baidu/vLLM-Kunlun
  ref: v0.25.1-dev
  commit: c53e090ff8800f586bf9e36e0d876779981bfb20
sources:
- repo://vllm_kunlun/models/__init__.py#L6-L54
- repo://vllm_kunlun/models/deepseek_v2.py#L472-L655
- repo://vllm_kunlun/v1/attention/backends/kunlun_attn.py
- repo://vllm_kunlun/models/qwen3_5.py#L84-L88
- repo://vllm_kunlun/lora/punica_wrapper/punica_kunlun.py#L70-L545
- repo://vllm_kunlun/lora/ops/kunlun_ops/lora_ops.py
- repo://vllm_kunlun/v1/structured_output/utils.py#L36-L119
- repo://docs/source/user_guide/support_matrix/supported_models.md
claims: .claims/model-support.json
---

# 模型支持

## 1. 四种接入机制

一个模型在 Kunlun 上跑起来可能依赖四类改动，读代码前先分清：

1. **完整 OOT 模型实现** —— `models/__init__.py#L14-L54` 里 10 个
   `ModelRegistry.register_model` 条目。
2. **[整模块重定向](architecture.md#41-整模块重定向7-个)** —— 7 个。
3. **[post-import 补丁](architecture.md#42-post-import-就地补丁8-个)** —— 8 个。
4. **[安装期 `cp` 覆盖](architecture.md#44-安装期文件覆盖最隐蔽)** —— 2 个。

> `models/__init__.py#L6` 有一行 `# TODO Remove all of models registration below`
> ——上游化是长期方向，这些 OOT 实现被视为临时状态。

## 2. 注册表

| 注册名 | 备注 |
| --- | --- |
| `Qwen3NextForCausalLM` | FLA/Mamba，见 [linear-attention.md](linear-attention.md) |
| `SeedOssForCausalLM` | 只为把 embedding 走到 OOT `VocabParallelEmbedding` |
| `MiMoV2FlashForCausalLM` | 普通 attention MoE，V head dim 不对称需 pad/slice |
| `GptOssForCausalLM` | attention sink + 交替滑动窗口 |
| `DeepseekV3ForCausalLM` | MLA |
| `DeepseekV32ForCausalLM` | **与 V3 同一个类**，靠 config 属性区分 |
| `DeepSeekMTPModel` | MTP draft |
| `GlmMoeDsaForCausalLM` | GLM MoE + DSA |
| `Qwen3_5MoeForConditionalGeneration` | 继承 qwen3_next |
| `Qwen3_5ForConditionalGeneration` | 同上 |

Gemma4 文本和多模态模型仍受支持，但不再注册 Kunlun 专用 OOT 模型，
而是直接使用上游 vLLM 的实现；设备差异由 Kunlun Attention 后端处理。

> ⚠️ `qwen3_dflash.py` 里的 `DFlashQwen3ForCausalLM` **不在这张表里**，
> 走不通正常注册流程。见 [spec-decode-and-sampling.md](spec-decode-and-sampling.md)。

## 3. 每个 OOT 实现存在的理由

| 模型文件 | 规模 | 为什么必须 OOT |
| --- | --- | --- |
| `deepseek_v2.py` | MLA + DSA 稀疏 indexer（包装成 `vllm::sparse_attn_indexer_vllm_kunlun`，`#L472-L655`） |
| `qwen3_next.py` | FLA / Mamba 算子替换 + `get_masked_input_and_mask_kunlun` |
| `qwen3_5.py` | 继承 qwen3_next，改成四路 QKVZ projection |
| `gpt_oss.py` | — | attention sink + 交替滑动窗口 |
| `mimo_v2_flash.py` | — | V 的 head dim 与 QK 不同，需要 pad/slice |
| `seed_oss.py` | — | 仅为把 embedding 路由到 OOT `VocabParallelEmbedding` |

Gemma4 原本在本地模型中处理的 Attention 缩放差异现已统一下沉到
`v1/attention/backends/kunlun_attn.py`：后端根据目标缩放计算 prefill kernel
的 `alpha`，上游模型无需再包含 Kunlun 专用缩放逻辑。

## 4. DeepSeek V3.2（DSA）的识别方式

**没有版本号判断，靠 `hasattr(config, "index_topk")`**，出现在三处：

- `platforms/kunlun.py#L235` —— `use_sparse`，决定 block size 强制
- `platforms/kunlun.py#L304-L310` —— 选 `FlashMLASparseBackend`
- 模型侧构建 indexer 层

含义：只要 HF config 里有 `index_topk`，就被当作稀疏 MLA 处理。
自定义 config 时要注意别误触发。

## 5. MTP

`models/deepseek_mtp.py`：

- `eh_proj` + 复用 `DeepseekV2DecoderLayer`
- `inputs_embeds[positions == 0] = 0`
- `current_step_idx = spec_step_idx % self.num_mtp_layers`

EAGLE proposer 里的 `mtp` 分支（`v1/sample/spec_decode/eagle.py#L116-L125`）见
[spec-decode-and-sampling.md](spec-decode-and-sampling.md)。

## 6. LoRA

**punica wrapper**：`lora/punica_wrapper/punica_kunlun.py`，
由 `KunlunPlatform.get_punica_wrapper` 返回（`platforms/kunlun.py#L383-L388`）。

**6 个算子**（`lora/ops/kunlun_ops/lora_ops.py`）：

| 阶段 | 算子族 |
| --- | --- |
| prefill | SGMV → `torch.ops.xspeedgate_ops.sgmv_*_sdnn` |
| decode | BGMV → `torch.ops.xspeedgate_ops.bgmv_*_cluster` |

实现细节：把 LoRA 当 MoE 风格描述符传下去，**硬编码 `expert_num = 9`**
（`#L442`，另见 `#L70`）；4-D 堆叠权重被 squeeze 成 3-D（`#L466`、`#L480`）。

> ⚠️ **两个调用点 arity 不匹配**：`add_lora_embedding`（`#L375-L399`）与
> `add_lora_logits`（`#L495-L545`）只传 4~5 个位置参数，而被调函数签名要 7~9 个。
> 这两条路径**一旦走到就会 TypeError**。

> ⚠️ **LoRA 只支持 V0 引擎**：`docs/source/user_guide/feature_guide/lora.md#L9`，
> 启动方式 `USE_ORI_ROPE=0 VLLM_USE_V1=0`。

## 7. 结构化输出（guided decoding）

`v1/structured_output/utils.py` 的修改是仓库里少数**记录充分且动机明确**的：

- `#L36` —— `_XPU_BACKEND = "torch_native"`，避免 `CUDA_ERROR_NOT_SUPPORTED`
- `#L112-L119` —— 用 `__dict__.get` 重新绑定，绕开 transformers 的
  ~192 个 alias 模块，注释给了实测代价：**每个进程约 0.75 s、约 38 MB**

## 8. 官方 tutorial 的实际启动命令

**8 篇 tutorial 全部是单机 TP-only**：没有任何一条命令包含
`--pipeline-parallel-size`、`--enable-expert-parallel` 或 `--data-parallel-size`。

### DeepSeek-V3.2-Exp-W8A8

```
--tensor-parallel-size 8
--dtype float16
--block-size 64
--max-model-len 32768
--max_num_seqs 32
--max_num_batched_tokens 8192
--no-enable-chunked-prefill
--no-enable-prefix-caching
--kv-cache-dtype bfloat16
--gpu-memory-utilization 0.95
```

量化通过**手改模型 `config.json`** 打开
（`docs/source/tutorials/multi_xpu_DeepSeek-V3.2-Exp-w8a8.md#L45`，
完整片段 `#L49-L95`），不是命令行 flag。见 [quantization.md](quantization.md)。

### GLM-5-W8A8-INT8

```
--tensor-parallel-size 8
--dtype bfloat16
--gpu-memory-utilization 0.97
--max_num_seqs 8
```

### Qwen3-Coder-480B-A35B（W8A8）

```
--tensor-parallel-size 8
--block-size 128
--max-model-len 40960
```

### LoRA

```
USE_ORI_ROPE=0 VLLM_USE_V1=0 ...
```

**启动前必须 `source setup_env.sh`**，见 [build-and-install.md](build-and-install.md)。

## 9. 惰性/未接线的模型相关代码

| 文件 | 状态 |
| --- | --- |
| `models/config.py` | **全部内容为注释**，包括一个 `fp8_ds_mla` 钩子 |
| `transformers_utils/config.py` | `_XPU_CONFIG_REGISTRY` 无引用方 |
| `transformers_utils/configs/qwen3_5*.py` | 未使用——模型在 `qwen3_5.py#L84-L88` import 的是上游 config |
| `tool_parsers/__init__.py#L8` | `TOOL_PARSERS = {}` 空字典，但 entry point 已接 |
| `reasoning/__init__.py#L8` | `REASONING_PARSERS = {}` 空字典，同上 |

后两条意味着 tool call / reasoning parser 的注册**通道已就位但没有内容**，
实际解析仍走上游默认实现。

## 10. 文档矩阵与实际的差异

- `docs/source/user_guide/support_matrix/supported_models.md` 只列 5 个模型；
  README 宣称 20+。
- `supported_features.md#L8-L14` 声称专家并行 🟢，但没有任何示例命令，
  且 EP 实现是 Python 逐专家循环（[moe-and-ep.md](moe-and-ep.md#5-ep-路径是-python-逐专家循环)）。

以代码和 tutorial 命令为准。完整冲突清单见 [known-gaps.md](known-gaps.md)。

## 相关页面

- [linear-attention.md](linear-attention.md) —— Qwen3-Next / Qwen3.5
- [attention-backend.md](attention-backend.md) —— MLA / 稀疏 MLA
- [quantization.md](quantization.md) —— W8A8 config.json 写法
- [build-and-install.md](build-and-install.md)
