# 📖 vLLM-Kunlun New Model Adaptation Manual

> Based on in-depth analysis of [baidu/vLLM-Kunlun](https://github.com/baidu/vLLM-Kunlun) and [vllm-project/vllm](https://github.com/vllm-project/vllm) repositories.
>
> Applicable Versions: vLLM v0.15.1+ / vLLM-Kunlun main branch

---

## Table of Contents

- [I. Understanding the Overall Architecture](#i-understanding-the-overall-architecture)
  - [1.1 Plugin System](#11-plugin-system)
  - [1.2 Startup Process](#12-startup-process)
  - [1.3 Import Hook Mechanism](#13-import-hook-mechanism)
  - [1.4 Code Architecture](#14-code-architecture)
- [II. New Model Adaptation Step-by-Step](#ii-new-model-adaptation-step-by-step)
  - [Step 0: Pre-assessment](#step-0-pre-assessment)
  - [Step 1: Implement Model Files](#step-1-implement-model-files)
  - [Step 2: Register the Model](#step-2-register-the-model)
  - [Step 3: Verify Registration](#step-3-verify-registration)
  - [Step 4: Testing](#step-4-testing)
- [III. Adaptation Guide for Special Model Types](#iii-adaptation-guide-for-special-model-types)
  - [3.1 MoE Models](#31-moe-models-eg-qwen3-moe-deepseek-v3)
  - [3.2 MLA Models](#32-mla-multi-latent-attention-models-eg-deepseek-v3)
  - [3.3 Multi-modal Models](#33-multi-modal-models-eg-qwen2-vl-internvl)
  - [3.4 Hybrid Attention Models](#34-hybrid-attention-models-eg-qwen3-next)
- [IV. Quantized Model Adaptation](#iv-quantized-model-adaptation)
  - [4.1 Supported Quantization Methods](#41-supported-quantization-methods)
  - [4.2 Special Handling for Quantization](#42-special-handling-for-quantization)
- [V. Custom Operators](#v-custom-operators-if-new-low-level-ops-are-needed)
- [VI. Common Pitfalls Checklist](#vi-common-pitfalls-checklist)
- [VII. Reference Template Quick Look-up](#vii-reference-template-quick-look-up)
- [VIII. Debugging Tips](#viii-debugging-tips)
- [IX. Environment Variables Cheat Sheet](#ix-environment-variables-cheat-sheet)
- [X. PR Submission Standards](#x-pr-submission-standards)

---

## I. Understanding the Overall Architecture

### 1.1 Plugin System

vLLM-Kunlun uses the **OOT (Out-of-Tree) Plugin** approach to integrate with vLLM, primarily registered via `entry_points` in `setup.py`:

```python
# setup.py
entry_points={
    'vllm.platform_plugins': ["kunlun = vllm_kunlun:register"],       # Platform Plugin
    'vllm.general_plugins': [
        "kunlun_model = vllm_kunlun:register_model",                   # Model Registration
        "kunlun_quant = vllm_kunlun:register_quant_method"             # Quantization Method
    ],
    "console_scripts": [
        "vllm_kunlun = vllm_kunlun.entrypoints.main:main"
    ]
}
```

### 1.2 Startup Process

```
vllm Startup
  ├─ 1. Discover platform_plugin → Call vllm_kunlun:register()
  │      ├─ Register KunlunPlatform (defines Attention Backend, Worker, etc.)
  │      ├─ Apply import hook (module redirection)
  │      └─ Register custom operators (custom_op)
  ├─ 2. Discover general_plugin → Call vllm_kunlun:register_model()
  │      └─ Register all Kunlun-adapted models via ModelRegistry.register_model()
  └─ 3. Model Loading → Match registered model classes based on the architectures field in config.json
```

### 1.3 Import Hook Mechanism

vLLM-Kunlun uses a custom import hook to **transparently replace** certain vLLM modules with Kunlun-customized versions:

```python
# vllm_kunlun/__init__.py
def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        module_mappings = {
            "vllm.compilation.wrapper":                        "vllm_kunlun.compilation.wrapper",
            "vllm.v1.worker.utils":                            "vllm_kunlun.v1.worker.utils",
            "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
            "vllm.v1.sample.ops.topk_topp_sampler":            "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
            "vllm.model_executor.layers.sampler":              "vllm_kunlun.ops.sample.sampler",
            "vllm.v1.sample.rejection_sampler":                "vllm_kunlun.v1.sample.rejection_sampler",
            "vllm.attention.ops.merge_attn_states":            "vllm_kunlun.ops.attention.merge_attn_states",
        }

        if module_name in module_mappings:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = module_mappings[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
    except Exception:
        pass

    return OLD_IMPORT_HOOK(module_name, globals=globals, locals=locals, fromlist=fromlist, level=level)
```

> **⚠️ Understanding this mechanism is crucial**: Even if you use `from vllm.xxx import YYY` in your model code, what you actually get might be `vllm_kunlun.xxx.YYY`.

### 1.4 Code Architecture

```
vllm_kunlun/
├── __init__.py                    # Plugin Entry: register() + import_hook()
├── platforms/kunlun.py            # KunlunPlatform: Defines Attention Backend, Worker, etc.
├── models/                        # ⭐ Model Implementation Directory (where you add files)
│   ├── __init__.py                # ⭐ Model Registration Entry
│   ├── deepseek_v2.py             # DeepSeek V2/V3 Reference Implementation
│   ├── deepseek_mtp.py            # DeepSeek MTP (Speculative Decoding)
│   ├── qwen3.py                   # Qwen3 Reference Implementation (Dense Model)
│   ├── qwen3_moe.py               # Qwen3 MoE Reference Implementation
│   ├── qwen3_next.py              # Qwen3-Next (Hybrid Attention)
│   ├── qwen3_vl.py                # Qwen3 VL (Multi-modal)
│   ├── qwen3_vl_moe.py            # Qwen3 VL MoE (Multi-modal + MoE)
│   ├── qwen2_vl.py                # Qwen2 VL
│   ├── qwen2_5_vl.py              # Qwen2.5 VL
│   ├── internlm2.py               # InternLM2 Reference Implementation
│   ├── internvl.py                # InternVL (Multi-modal)
│   ├── interns1.py                # InternS1
│   ├── seed_oss.py                # SeedOss
│   ├── gpt_oss.py                 # GptOss
│   └── mimo_v2_flash.py           # MiMo-V2-Flash
├── ops/                           # Kunlun Custom Operators
│   ├── _kunlun_ops.py             # KunlunOps: paged_attention, rms_norm, silu...
│   ├── _custom_ops.py             # vllm custom_op registration
│   ├── activation.py              # Activation functions like SiluAndMul, GeluAndMul
│   ├── attention/                 # Attention Operators
│   │   ├── layer.py               # Attention Layer Wrapper
│   │   └── backends/kunlun_attn.py # KunlunAttentionBackend + KunlunAttentionImpl
│   ├── quantization/              # Quantization related: AWQ, GPTQ, CompressedTensors...
│   ├── vocab_parallel_embedding.py # Custom Embedding
│   └── rotary_embedding.py        # Split_Norm_Rope (QKNorm + RoPE Fusion)
├── v1/attention/backends/         # Attention Backend for v1 Engine
│   ├── kunlun_attn.py             # Standard Attention
│   └── mla/                       # MLA (Multi-Latent Attention) Implementation
│       ├── flashmla.py
│       ├── flashmla_sparse.py
│       └── common.py
├── compilation/wrapper.py         # torch.compile Wrapper
├── config/                        # Model Configuration Overrides
│   └── model.py                   # Patch for attributes like is_deepseek_mla
├── distributed/                   # Communication related
│   └── kunlun_communicator.py     # Kunlun Device Communication
└── csrc/                          # C++ Extensions
    └── utils.cpp
```

---

## II. New Model Adaptation Step-by-Step

### Step 0: Pre-assessment

Before starting, confirm which scenario your model falls into:

| Scenario | Description | Effort |
|------|------|--------|
| **Case A: vLLM already supports the model** | Only need to replace Attention / Activation with Kunlun versions | ⭐ Minimal |
| **Case B: vLLM does not support, new architecture needed** | Requires full implementation of model class + registration | ⭐⭐⭐ High |
| **Case C: MoE variant of an existing model** | Add MoE layer on top of the Dense version | ⭐⭐ Medium |
| **Case D: Multi-modal model** | Language Model + Vision Encoder + Projector | ⭐⭐⭐⭐ Maximum |

**Recommended Workflow:**

1. Check the [vLLM Supported Models List](https://docs.vllm.ai/en/stable/models/supported_models.html) to see if the model is already there.
2. If yes → Copy the corresponding file from `vllm/model_executor/models/` to `vllm_kunlun/models/` and perform replacements.
3. If no → Refer to the [vLLM Adding a New Model Documentation](https://docs.vllm.ai/en/stable/contributing/model/) to understand the principles first, then follow this manual.

---

### Step 1: Implement Model Files

Create a model file in the `vllm_kunlun/models/` directory, e.g., `my_new_model.py`.

#### 1.1 Key Replacement Comparison Table

| Component | vLLM Native Import | vLLM-Kunlun Replacement Import | Required? |
|------|-----------------|------------------------|---------|
| **Attention Layer** | `from vllm.attention import Attention` | `from vllm_kunlun.ops.attention.layer import Attention` | ✅ **Yes** |
| **SiluAndMul** | `from vllm.model_executor.layers.activation import SiluAndMul` | `from vllm_kunlun.ops.activation import SiluAndMul` | ✅ **Yes** |
| **GeluAndMul** | `...activation import GeluAndMul` | `from vllm_kunlun.ops.activation import GeluAndMul` | ⚠️ As needed |
| **QuickGELU** | `...activation import QuickGELU` | `from vllm_kunlun.ops.activation import QuickGELU` | ⚠️ As needed |
| **VocabParallelEmbedding** | `from vllm...vocab_parallel_embedding import VocabParallelEmbedding` | `from vllm_kunlun.ops.vocab_parallel_embedding import VocabParallelEmbedding` | ⚠️ Some models |
| **ParallelLMHead** | Same as above | `from vllm_kunlun.ops.vocab_parallel_embedding import ParallelLMHead` | ⚠️ Some models |
| **RoPE (Special)** | `from vllm...rotary_embedding import get_rope` | `from vllm_kunlun.ops.rotary_embedding import Split_Norm_Rope` | ⚠️ MoE+QKNorm |
| **Linear / RMSNorm, etc.** | Use vLLM native directly | **No replacement needed** | — |

> 💡 **Core Principle**: Any component involving **CUDA kernel calls** (Attention, Activation, Sampling) must be replaced with the Kunlun version; pure PyTorch components (Linear, RMSNorm, RoPE, etc.) can use vLLM native directly.

#### 1.2 Standard Dense Decoder-Only Model Template

Refer to `qwen3.py` or `internlm2.py`:

```python
"""Inference-only MyNewModel compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import MyNewModelConfig  # HuggingFace config

# ==========================================
# ⭐ Key Replacement 1: Use Kunlun-customized Attention
# ==========================================
# Do not use from vllm.attention import Attention
from vllm_kunlun.ops.attention.layer import Attention

# ==========================================
# ⭐ Key Replacement 2: Use Kunlun-customized Activation
# ==========================================
# Do not use from vllm.model_executor.layers.activation import SiluAndMul
from vllm_kunlun.ops.activation import SiluAndMul

# Other layers can use vLLM native directly
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear, RowParallelLinear, MergedColumnParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsPP, SupportsLoRA
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, extract_layer_index,
    is_pp_missing_parameter, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix
)


# ============================
# 1. MLP Layer
# ============================
class MyNewModelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act,
                 quant_config=None, prefix=""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()  # ⭐ Use Kunlun version

    def forward(self, x):
        # Implementation...
```

#### 1.3 Key Implementation Requirements

- **All modules must include the `prefix` parameter**, passed in `__init__()`.
- **`@support_torch_compile` decorator** must be added to the main model class (e.g., `MyNewModel`).
- **`load_weights()` method** must correctly handle weight name mapping (`stacked_params_mapping`).
- **Pipeline Parallelism (PP)** requires using tools like `PPMissingLayer`, `is_pp_missing_parameter`, etc.

---

## Step 2: Register the Model

Add registration code in `vllm_kunlun/models/__init__.py`:

```python
# vllm_kunlun/models/__init__.py

from vllm import ModelRegistry

def register_model():
    # ... Existing model registrations ...

    # ⭐ Add your new model (using lazy loading string format)
    ModelRegistry.register_model(
        "MyNewModelForCausalLM",                                    # ← Must match architectures in config.json
        "vllm_kunlun.models.my_new_model:MyNewModelForCausalLM"    # ← Module path:Class name
    )
```

**⚠️ Key Considerations:**

1. The **first parameter** of `register_model()` is the model's `architecture` identifier, which **must exactly match the `"architectures"` field in the HuggingFace model's `config.json`**.

2. Use the **string format** for the module path (`"module:class"`) to implement **lazy loading**, avoiding CUDA initialization conflicts (`RuntimeError: Cannot re-initialize CUDA in forked subprocess`).

3. If the model already exists in vLLM (e.g., `Qwen3ForCausalLM`), the Kunlun version will **overwrite** the original vLLM version upon registration.

---

## Step 3: Verify Registration

### Case A: Overwriting an Existing vLLM Model Architecture

If your model architecture name (e.g., `"Qwen3ForCausalLM"`) already exists in vLLM, vLLM will output the following log during registration:

```
WARNING [...] Model architecture Qwen3ForCausalLM is already registered,
and will be overwritten by the new model class
vllm_kunlun.models.qwen3:Qwen3ForCausalLM.
```

Seeing this log indicates a successful overwrite ✅.

### Case B: Brand New Model Architecture

If you are registering an architecture that does not exist in vLLM, there is no default log confirmation. It is recommended to verify manually during the debugging phase:

```python
from vllm import ModelRegistry
assert "MyNewModelForCausalLM" in ModelRegistry.get_supported_archs()
print("✅ Model registration successful!")
```

---

## Step 4: Testing

### 4.1 Offline Inference Test

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/MyNewModel",
    trust_remote_code=True,
    dtype="float16",
    tensor_parallel_size=1,  # Verify with single card first
)

outputs = llm.generate(
    ["Hello, please introduce yourself."],
    SamplingParams(temperature=0.7, max_tokens=256),
)
for output in outputs:
    print(output.outputs[0].text)
```

#### 4.2 Online Service Test

```bash
XPU_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8888 \
    --model /path/to/MyNewModel \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 4096 \
    --block-size 64
```

#### 4.3 Accuracy Verification

It is recommended to compare results with HuggingFace Transformers CPU/GPU inference:

```python
# Transformers reference output
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/path/to/MyNewModel", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("/path/to/MyNewModel")
# ... Generate and compare output
```

---

## III. Adaptation Guide for Special Model Types

### 3.1 MoE Models (e.g., Qwen3-MoE, DeepSeek-V3)

**Reference Files:**
- `vllm_kunlun/models/qwen3_moe.py`
- `vllm_kunlun/models/deepseek_v2.py`

**Additional Points:**

- Use `vllm.model_executor.layers.fused_moe.layer.FusedMoE`; Kunlun has replaced the underlying kernel via import hook.
- MoE's `load_weights()` is more complex, requiring expert parameter mapping:

```python
expert_params_mapping = FusedMoE.make_expert_params_mapping(
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=config.n_routed_experts,
)
```

- Recommended environment variables:

```bash
export KUNLUN_USE_MOE_FFN_BLOCK=True
export XPU_USE_MOE_SORTED_THRES=120
```

### 3.2 MLA (Multi-Latent Attention) Models (e.g., DeepSeek-V3)

**Reference File:** `vllm_kunlun/models/deepseek_v2.py`

**MLA Special Handling:**
- KV compression dimensions: `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`.
- Platform layer automatically selects `FlashMLABackend`:

```python
# vllm_kunlun/platforms/kunlun.py
if use_mla:
    if use_sparse:
        return "vllm_kunlun.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    return "vllm_kunlun.v1.attention.backends.mla.flashmla.FlashMLABackend"
```

- `block_size` usually needs to be set to **64**.
- Recommended setting: `export USE_ORI_ROPE=1`.

### 3.3 Multi-modal Models (e.g., Qwen2-VL, InternVL)

**Reference Files:**
- `vllm_kunlun/models/qwen3_vl.py`
- `vllm_kunlun/models/internvl.py`
- `vllm_kunlun/models/interns1.py`

**Additional Components to Implement:**

| Component | Description |
|------|------|
| `SupportsMultiModal` Interface | Declares that the model supports multi-modal input |
| Vision Encoder | Usually `InternVisionModel` or custom ViT |
| Projector | Vision → Language mapping (e.g., MLP) |
| `@MULTIMODAL_REGISTRY.register_processor(...)` | Register multi-modal processor |
| `BaseMultiModalProcessor` | Handles multi-modal input |
| `BaseProcessingInfo` | Handles processing info |
| `BaseDummyInputsBuilder` | Dummy inputs for the profiling phase |

### 3.4 Hybrid Attention Models (e.g., Qwen3-Next)

**Reference File:** `vllm_kunlun/models/qwen3_next.py`

This model contains both **Linear Attention** and **Full Attention** layer types:

```python
# Select different attention calculations based on layer_type
if self.layer_type == "linear_attention":
    self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
elif self.layer_type == "full_attention":
    self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)
```

Note:
- Linear Attention uses `GatedDeltaNet` or similar implementations.
- Need to register custom `custom_op` (e.g., `vllm.gdn_attention`) for `splitting_ops` in `torch.compile`.

---

## IV. Quantized Model Adaptation

### 4.1 Supported Quantization Methods

| Quantization Method | Adaptation File | Status |
|---------|---------|------|
| **INT8 Dynamic (W8A8)** | `ops/quantization/kernels/kunlun_scale_mm.py` | ✅ Recommended |
| **AWQ (INT4)** | `ops/quantization/awq.py` | ✅ Supported |
| **GPTQ (INT4)** | `ops/quantization/gptq.py` | ✅ Supported |
| **CompressedTensors (INT8 MoE)** | `ops/quantization/compressed_tensors/` | ✅ Supported |
| **FP8** | — | ⚠️ Partial Support |
| **bfloat16** | — | ⚠️ Double VRAM bug |

### 4.2 Special Handling for Quantization

Kunlun chips use the **max value** for scale calculation instead of vLLM's default absmax:

```python
# ops/quantization/kernels/kunlun_scale_mm.py
class KunlunScaledMMLinearKernel(CutlassScaledMMLinearKernel):
    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        # ⭐ Key: Multiply scale by 127.0 to convert to max format
        with torch.no_grad():
            getattr(layer, self.w_s_name).mul_(127.0)
```

INT4 weights need to be **repacked** into the Kunlun layout order:

```python
# AWQ repack example
AWQ_TO_KUNLUN_ORDER_NORMAL = [4, 0, 5, 1, 6, 2, 7, 3]
unpacked_kunlun = unpacked_awq[..., AWQ_TO_KUNLUN_ORDER_NORMAL]
```

---

## V. Custom Operators (if new low-level Ops are needed)

If your model requires new low-level operators:

### 5.1 Wrap kunlun_ops calls in `_kunlun_ops.py`

```python
# vllm_kunlun/ops/_kunlun_ops.py
class KunlunOps:
    @staticmethod
    def my_new_op(input, weight, out):
        """Call underlying kunlun_ops implementation"""
        kunlun_ops.my_new_op(input, weight, out=out)
```

### 5.2 Register to vLLM in `_custom_ops.py`

Follow the **three-piece pattern**:

```python
# vllm_kunlun/ops/_custom_ops.py

# 1. Define the actual implementation of the op
def my_new_op_impl(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    KunlunOps.my_new_op(input, weight, output)
    return output

# 2. Define fake tensor implementation (for torch.compile)
def my_new_op_fake(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)

# 3. Register
direct_register_custom_op(
    op_name="my_new_op",
    op_func=my_new_op_impl,
    mutates_args=[],
    fake_impl=my_new_op_fake,
)
```

---

## VI. Common Pitfalls Checklist

Before submitting a PR, please check each item:

- [ ] **Attention** uses `vllm_kunlun.ops.attention.layer.Attention`?
- [ ] **Activation functions** use `vllm_kunlun.ops.activation.SiluAndMul`, etc.?
- [ ] All submodules in `__init__()` have the `prefix` parameter passed?
- [ ] `load_weights()` correctly handles weight name mapping (`stacked_params_mapping`)?
- [ ] `@support_torch_compile` decorator is added to the main model class?
- [ ] The first parameter of `ModelRegistry.register_model()` exactly matches `architectures` in `config.json`?
- [ ] No use of `VLLM_USE_V1` environment variable for logic (deprecated, v0.15.1 is V1-only)?
- [ ] Type annotations use `Optional[T]` instead of `T | None` (to avoid `infer_schema` failure)?
- [ ] Quantized model scales are correctly multiplied by `127.0`?
- [ ] Supports Pipeline Parallelism (using `PPMissingLayer`, `is_pp_missing_parameter`)?
- [ ] Ran `pre-commit` format checks?
- [ ] Commits use `-s` signature (DCO compliance)?

---

## VII. Reference Template Quick Look-up

| Model Type | Best Reference File | Features |
|---------|------------|------|
| Standard Dense LLM | `qwen3.py` | Simplest, recommended for beginners |
| Dense LLM (Custom Embedding) | `seed_oss.py`, `internlm2.py` | Custom VocabParallelEmbedding |
| MoE LLM | `qwen3_moe.py` | FusedMoE + EP + SharedExpert |
| MLA + MoE (DeepSeek) | `deepseek_v2.py` | MLA attention + MoE + Indexer |
| Hybrid Attention | `qwen3_next.py` | Linear + Full attention |
| Multi-modal (VL) | `qwen3_vl.py`, `internvl.py` | ViT + Projector + LLM |
| Speculative Decoding (MTP) | `deepseek_mtp.py` | Multi-Token Prediction |

---

## VIII. Debugging Tips

### 8.1 Startup Failure

- **`ModuleNotFoundError`**: Check if the import hook mapping table in `__init__.py` covers the corresponding module.
- **`circular import`**: Check if your new code introduces heavy dependencies during the `register()` phase.
- **`Model architecture XXX is not supported`**: Check if the first parameter of `register_model()` matches `config.json`.

### 8.2 Abnormal Output

- **Garbage output**: Compare with HF transformers output on CPU; likely an operator precision issue or weight loading mapping error.
- **Repeated tokens**: Check if `rotary_embedding` is applied correctly and if the `is_neox_style` parameter is correct.
- **Truncated output**: Check `max_model_len` settings and if KV cache is sufficient.

### 8.3 VRAM Issues

- Use `--dtype float16` (avoid bfloat16 due to double VRAM bug).
- Set `VLLM_KUNLUN_ENABLE_INT8_BMM=1` (saves ~0.1GB).
- Lower `--gpu-memory-utilization` (default is 0.9).
- Use INT8 quantized models.

### 8.4 Weight Loading Failure

```python
# Debugging method: Print parameter names for comparison
params_dict = dict(self.named_parameters())
print("=== Model params ===")
for k in sorted(params_dict.keys()):
    print(f"  {k}: {params_dict[k].shape}")

# Print in load_weights
for name, loaded_weight in weights:
    if name not in params_dict:
        print(f"  ⚠️ Skipped: {name}")
```

### 8.5 Kunlun Graph Failure

Confirm that `splitting_ops` in `compilation-config` includes your attention op name:

```json
{
  "splitting_ops": [
    "vllm.unified_attention",
    "vllm.unified_attention_with_output",
    "vllm.unified_attention_with_output_kunlun",
    "vllm.sparse_attn_indexer_vllm_kunlun"
  ],
  "cudagraph_mode": "PIECEWISE"
}
```

---

## IX. Environment Variables Cheat Sheet

```bash
# === Required ===
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # Specify Kunlun cards to use
export VLLM_HOST_IP=$(hostname -i)             # IP for distributed communication

# === Recommended ===
export XMLIR_FORCE_USE_XPU_GRAPH=1             # Enable XPU Graph acceleration
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false   # Disable mock compile
export XMLIR_CUDNN_ENABLED=1                   # Enable cuDNN equivalent acceleration
export XPU_USE_DEFAULT_CTX=1                   # Default context
export BKCL_FORCE_SYNC=1                       # BKCL forced sync (multi-card stability)

# === Model Specific ===
export USE_ORI_ROPE=1                          # DeepSeek series uses original RoPE
export XFT_USE_FAST_SWIGLU=1                   # Fast SwiGLU activation
export XPU_USE_FAST_SWIGLU=1                   # Same as above (some versions)
export XPU_USE_MOE_SORTED_THRES=120            # MoE sorting threshold
export KUNLUN_USE_MOE_FFN_BLOCK=True           # MoE FFN block optimization

# === Optional Tuning ===
export VLLM_KUNLUN_ENABLE_INT8_BMM=1           # Enable INT8 BMM (saves ~0.1GB)
```

---

## X. PR Submission Standards

### 10.1 Branch Naming

```
feature/add-my-new-model
bugfix/fix-attention-output
```

### 10.2 Commit Message Prefix

| Prefix | Description |
|------|------|
| `[Feature]` | New functionality / New model |
| `[Bugfix]` | Bug fix |
| `[CI/Build]` | CI / Build related |
| `[Doc]` | Documentation update |
| `[Misc]` | Others |

### 10.3 Before Submission

```bash
# 1. Install pre-commit
pre-commit install

# 2. Run checks
pre-commit run --all-files

# 3. Signed commit (DCO compliance)
git commit -s -m "[Feature] Add MyNewModel support for Kunlun"
```

### 10.4 PR Checklist

- [ ] Code passes `pre-commit` checks.
- [ ] Single-card offline inference test passed.
- [ ] Multi-card TP test passed (if applicable).
- [ ] Quantized model test passed (if applicable).
- [ ] Updated `vllm_kunlun/models/__init__.py` registration.
- [ ] Updated supported models list in README (if applicable).

---

## Appendix: Standard Startup Command Templates

### A. Standard Dense Model (Single Card)

```bash
XPU_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8888 \
    --model /path/to/model \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 8192 \
    --block-size 64
```

### B. MoE Model (8-card TP)

```bash
XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XMLIR_FORCE_USE_XPU_GRAPH=1 \
KUNLUN_USE_MOE_FFN_BLOCK=True \
XPU_USE_MOE_SORTED_THRES=120 \
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8888 \
    --model /path/to/moe-model-int8 \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --max_num_seqs 4 \
    --block-size 64 \
    --no-enable-chunked-prefill \
    --distributed-executor-backend mp \
    --no-enable-prefix-caching
```

### C. DeepSeek-V3 (MLA + MoE, W8A8)

```bash
XMLIR_ENABLE_MOCK_TORCH_COMPILE=false \
USE_ORI_ROPE=1 \
XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8806 \
    --model /path/to/DeepSeek-V3-w8a8 \
    --gpu-memory-utilization 0.98 \
    --trust-remote-code \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --dtype float16 \
    --max_num_seqs 4 \
    --block-size 64 \
    --no-enable-chunked-prefill \
    --distributed-executor-backend mp \
    --no-enable-prefix-caching
```

---

> 📝 **Document Maintenance**: If you have questions or suggestions, please provide feedback in [GitHub Issues](https://github.com/baidu/vLLM-Kunlun/issues).
