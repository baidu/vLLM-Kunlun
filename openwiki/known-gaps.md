---
type: reference
title: 已知缺口、死代码与文档冲突
summary: >-
  PD 分离不支持的穷尽证据、名字与行为不符的函数、无消费者的环境变量与死代码、
  以及代码与文档之间的七处冲突。
generated:
  by: hand-authored (Claude Code, OpenWiki OKF v0.2 conventions)
  at: 2026-09-02T00:00:00Z
evidence_version:
  repo: https://github.com/baidu/vLLM-Kunlun
  ref: v0.25.1-dev
  commit: c53e090ff8800f586bf9e36e0d876779981bfb20
sources:
- repo://vllm_kunlun/ops/attention/layer.py#L9-L232
- repo://docs/envs.py#L94-L151
- repo://vllm_kunlun/ops/attention/flashmla.py#L34-L58
- repo://vllm_kunlun/v1/worker/utils.py#L110-L112
- repo://vllm_kunlun/quantization/moe_wna16.py#L27-L107
- repo://vllm_kunlun/models/config.py
- repo://vllm_kunlun/platforms/envs.py#L34-L71
- repo://pyproject.toml#L6-L19
claims: .claims/known-gaps.json
---

# 已知缺口、死代码与文档冲突

本页记录**代码与文档冲突、死代码、名字骗人的符号、以及明确不支持的特性**。
[INSTRUCTIONS.md](INSTRUCTIONS.md) 要求这类冲突单独成条，以代码为准。

## 1. PD 分离（Prefill/Decode Disaggregation）不支持

这是**最容易被误判**的一条，所以给出穷尽搜索的否定证据：

| 检查项 | 结果 |
| --- | --- |
| KV connector 实现文件 | `vllm_kunlun/distributed/` 下**没有**任何 connector |
| `nixl` 引用 | 全仓库 **0** 处 |
| `--kv-transfer-config` | 全仓库 **0** 处 |
| 特性矩阵 | **未列出** PD 分离 |
| 唯一相关代码 | `ops/attention/layer.py#L9-L13`、`#L191-L232` —— 从上游继承的通用 KV connector 钩子点，**没有 connector group 时全部 no-op** |

**化石证据**（说明曾经有过尝试，但已废弃）：

- `docs/envs.py#L94-L113` —— 死代码里有
  `DISAGGREGATED_PREFILL_RANK_TABLE_PATH`、`LLMDataDistCMgrConnector`、
  `VLLM_KUNLUN_LLMDD_RPC_IP` / `_PORT`。这个文件整体是从 vllm-ascend
  抄来的，**没有任何 import 方**。
- `docs/envs.py#L149-L151` —— mooncake 相关注释。
- `supported_features.po#L175` —— 废弃的 gettext 条目
  `#~ msgid "Prefill Decode Disaggregation"`（`#~` 前缀表示该条已从源文档移除）。
- `faqs.po#L251-L252` —— 同类废弃条目。

> **易误判点**：tutorial 的 curl 示例里出现 `kv_transfer_params: null`。
> 这是 vLLM 请求体的通用字段，**不代表支持 PD 分离**。

## 2. 名字与行为不符的符号

读代码时最容易被骗的四处：

| 符号 | 名字暗示 | 实际行为 | 位置 |
| --- | --- | --- | --- |
| `is_flashmla_supported()` | 能力探测 | **永远** `return True, None` | `ops/attention/flashmla.py#L34-L38` |
| `get_mla_metadata()` | 返回 tile scheduler metadata | 返回 `cache_seqlens_cpu, cache_seqlens` | `ops/attention/flashmla.py#L56-L58` |
| `flashinfer_sample()` + `"Using FlashInfer..."` 日志 | 用了 FlashInfer | 调 `kunlun_ops.*`；同文件 `#L15-L17` 明确 FlashInfer 不支持 | `v1/sample/ops/topk_topp_sampler.py#L28-L34`、`#L206-L219` |
| `KVBlockZeroer.zero_block_ids` | 把 block 清零 | 空实现 `return` | `v1/worker/utils.py#L110-L112` |

第二条的连带后果：FlashMLA builder 里 CUDA-graph 持久 buffer 那段被整块注释
（`v1/attention/backends/mla/flashmla.py#L113-L138`），但 `(num_sms, 8)` 的
buffer 仍在分配（`#L81-L96`）。

另外 `MiMoV2Flash` 名字里的 "Flash" **不代表线性注意力/FlashAttention**，
它是普通 attention MoE（`models/mimo_v2_flash.py#L190`、`#L639`）。

## 3. 死代码与未接线代码

| 位置 | 状态 |
| --- | --- |
| `vllm_kunlun/utils.py` | **没有任何模块 import 它**，其中的日志/hook 消费者因此永不执行 |
| `docs/envs.py` | 从 vllm-ascend 抄来（`#L5` 自认），无 import 方 |
| `quantization/moe_wna16.py#L27` | import 路径不存在（`vllm_kunlun.ops.quantization.kernels.quant_ops`）；`#L107` 的 `KunlunMoeWNA16Method` 无引用 |
| `models/config.py` | **全部内容为注释**，含一个 `fp8_ds_mla` 钩子 |
| `transformers_utils/config.py` | `_XPU_CONFIG_REGISTRY` 无引用方 |
| `transformers_utils/configs/qwen3_5*.py` | 未使用（模型在 `qwen3_5.py#L84-L88` import 上游 config） |
| `ops/native_ops.py` | 纯 PyTorch 对照实现，无 import 方（诊断用，见 [linear-attention.md](linear-attention.md#8-纯-pytorch-调试旁路)） |
| `ops/fla/chunk.py#L297-L321` | `if False:` 死分支 |
| `ops/_kunlun_ops.py#L478-L496` | `M * top_k < 400` 小 batch 路径整块注释 |
| `compressed_tensors_moe.py#L109-L113` | W4A8 分支整块注释 |
| `tool_parsers/__init__.py#L8` | `TOOL_PARSERS = {}` 空注册表（entry point 已接） |
| `reasoning/__init__.py#L8` | `REASONING_PARSERS = {}` 空注册表（entry point 已接） |
| `models/qwen3_dflash.py` 的 `DFlashQwen3ForCausalLM` | 未在 `models/__init__.py` 注册 |
| `pyproject.toml#L18-L19` | console script 指向不存在的 `vllm_kunlun.cmdline` |

## 4. 环境变量：9 个里 7 个是死的

`platforms/envs.py#L34-L71` 定义 9 个变量，**只有
`VLLM_KUNLUN_ENABLE_INT8_BMM` 有真实消费者**（`mla/common.py#L1152`、
`#L1459`、`#L1969`）。

- 6 个完全没有消费者：`ENABLE_VLLM_INFER_HOOK`、`ENABLE_VLLM_OPS_HOOK`、
  `ENABLE_VLLM_MODULE_HOOK`、`ENABLE_VLLM_MOE_FC_SORTED`、
  `ENABLE_CUSTOM_DPSK_SCALING_ROPE`、`ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE`
- 2 个（`VLLM_MULTI_LOGPATH`、`ENABLE_VLLM_MULTI_LOG`）的消费者在
  `vllm_kunlun/utils.py`，而该模块无人 import
- `VLLM_MULTI_LOGPATH` 的 TYPE_CHECKING 存根（`#L7`）写 `"./log"`，
  实际默认值是 `"./logs"`

反过来，有一个**真实生效但既不在这个列表里也不在文档里**的变量：
`FAST_RANDOM_SAMPLE`（`v1/sample/ops/topk_topp_sampler.py#L161-L167`，
直接 `os.getenv` 读）。

详见 [architecture.md](architecture.md#7-环境变量两套互不相交的集合)。

## 5. 代码与文档的七处冲突

| # | 冲突 | 以哪个为准 |
| --- | --- | --- |
| 1 | README 说构建后端是 hatchling；`pyproject.toml#L6` 是 setuptools | 代码 |
| 2 | CI 里 pin `vllm==0.11.0`，插件版本是 0.25.1 | `faqs.md#L43` 的规则：版本必须相同 → 0.25.1 |
| 3 | Dockerfile `transformers==4.57.1` vs `requirements.txt` `5.2.0` | requirements |
| 4 | 默认分支是 `v0.25.1-dev`，但 CONTRIBUTING / `conf.py` / workflows 都写 `main` | `v0.25.1-dev` |
| 5 | `multi_xpu_GLM-4.5.md` 标题写 "Single XPU"，命令是 TP=8 | 命令 |
| 6 | `supported_models.md` 列 5 个模型，README 宣称 20+ | `models/__init__.py` 的 10 个注册项 |
| 7 | `supported_features.md#L8-L14` 声称专家并行 🟢，但 0 条示例命令，且 EP 是 Python 逐专家循环 | 代码（见 [moe-and-ep.md](moe-and-ep.md#5-ep-路径是-python-逐专家循环)） |

另外两处版本号自相矛盾：`platforms/version.py#L3` = `0.25.1` vs
`pyproject.toml#L10` = `0.25.1.dev0`；以及 `main` 分支的 README 仍宣称
v0.15.1 / "Initial release"。

## 6. 明确未实现 / 抛异常的特性

| 特性 | 状态 | 证据 |
| --- | --- | --- |
| PD 分离 | 不支持 | 见本页第 1 节 |
| cascade attention | 无条件不支持 | `kunlun_attn.py#L992-L1013` `return False` |
| MLA + KV sharing | `NotImplementedError` | `mla/common.py#L1024-L1025` |
| 混合 Mamba 模型 + 投机解码 | `NotImplementedError`（需实现 `MambaSpecDecodeGPUContext`） | `v1/worker/mamba_utils.py#L277-L281` |
| `CUDAGraphMode.FULL` / `FULL_AND_PIECEWISE` | 不支持 | `Kunlun_Graph.md#L74-L75` |
| `use_inductor` | 不支持 | 同上；`platforms/kunlun.py#L276-L279` 强制 `backend="eager"` |
| W4A8 compressed-tensors MoE | 注释掉 | `compressed_tensors_moe.py#L109-L113` |
| AWQ MoE 的 Kunlun 实现 | 回落上游 `MoeWNA16Config` | `quantization/awq.py#L58-L70` |
| DFlash 并行 drafting + 预插 context K/V | stub | `models/qwen3_dflash.py#L36-L45` |
| LoRA V1 引擎 | 只支持 V0 | `lora.md#L9` |
| fp8 权重/激活 | dtype 白名单不含 fp8 | `platforms/kunlun.py#L390-L405` |

## 7. 潜在 bug（未修）

| 问题 | 位置 |
| --- | --- |
| `opaque_attention_op` 缺 `@classmethod` | `platforms/kunlun.py#L407-L411` |
| `KunlunCommunicator.change_state` 在 `finally` 之外恢复状态，异常会泄漏 | `distributed/kunlun_communicator.py#L66-L86` |
| `add_lora_embedding` 传 4~5 个位置参数，签名要 7~9 个 | `lora/punica_wrapper/punica_kunlun.py#L375-L399` |
| `add_lora_logits` 同上 | 同文件 `#L495-L545` |
| LoRA 硬编码 `expert_num = 9` | 同文件 `#L442`、`#L70` |
| `paged_attention_v2` 与 `v1` 函数体完全相同 | `ops/_kunlun_ops.py#L50-L131` |
| `check_and_update_config` 里 worker_cls 的两个分支赋同一个值 | `platforms/kunlun.py#L217-L223` |
| `check_and_update_config` docstring 仍写 `TODO Update here for v0.15.1` | `platforms/kunlun.py#L183-L184` |
| `get_attn_backend_cls` docstring 已过期 | `platforms/kunlun.py#L293-L299` |
| `.pre-commit-config.yaml#L75` 排除了 actionlint 唯一的检查对象 | — |

## 相关页面

- [architecture.md](architecture.md)
- [platform-contract.md](platform-contract.md)
- [testing-and-ci.md](testing-and-ci.md)
- [INSTRUCTIONS.md](INSTRUCTIONS.md)
