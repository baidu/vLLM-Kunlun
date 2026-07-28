# DeepSeek V4 Flash on Kunlun

## Current Status

- Phase: 3, basic-layer dummy-forward bring-up
- Target architecture: `DeepseekV4ForCausalLM`
- vLLM: `0.25.1`
- Kunlun package: `0.21.0.dev0`
- Model: `/mnt/cluster/models/DeepSeek-V4-Flash-Base`
- Tensor parallelism: 4
- Cards: `XPU_VISIBLE_DEVICES=0,1,2,3`
- New launcher: `/mnt/cluster/wangtianyu/shell/service_deepseek_v4_4gpu.sh`
- Validation status: all 46 checkpoint shards load on four ranks; dummy forward crosses FP8 routed experts and reaches the Lightning Indexer fused Q/RoPE/FP8 quantization boundary

The existing `/mnt/cluster/wangtianyu/shell/service.sh` is preserved. It contains Qwen-specific service arguments and must not be used as the DeepSeek V4 baseline.

## Phase 0 Result

The first four-card probe reached:

```text
Resolved architecture: DeepseekV4ForCausalLM
```

It then failed before model construction because the Kunlun platform code accessed `vllm.envs.VLLM_ATTENTION_BACKEND`, which is not present in vLLM 0.25.1. The minimal compatibility fix reads the same environment variable with `os.getenv`. The failure is classified as `config/platform compatibility`, not a V4 model failure.

The probe also reports that Proteus auto-install cannot run because the container Python has no `pip`. This environment issue is recorded separately from the vLLM failure.

The first platform compatibility fixes are kept small and isolated in `vllm_kunlun/platforms/kunlun.py`:

- read `VLLM_ATTENTION_BACKEND` with `os.getenv` because the 0.25.1 `vllm.envs` module no longer exposes it;
- import `is_flashmla_supported` from the existing Kunlun ops package;
- retain the `vllm.envs` import because later compatibility checks still use `getattr(envs, ...)`.

After these fixes, the API process passed platform config validation and spawned four worker processes. All workers reached `DeepseekV4ForCausalLM` construction through the community `vllm/models/deepseek_v4/nvidia/model.py` path.

The first model-level blocker was:

```text
choose_scaled_mm_linear_kernel
platform_kernels = possible_kernels[current_platform._enum]
KeyError: <PlatformEnum.OOT: 6>
```

This was classified as `custom-op dispatch / FP8 kernel routing`. The fix added ordinary and block FP8 OOT registries, deferred registration until the vLLM linear module was fully imported, and added thin Kunlun wrappers that reuse the upstream Cutlass implementation while overriding only the platform support check.

The four workers now select `KunlunFP8ScaledMMLinearKernel` and `KunlunFp8BlockScaledMMKernel`, finish model construction, and start loading the 274.44 GiB checkpoint (46 safetensors shards).

The launcher also sets:

- `--kv-cache-dtype fp8`, required by the V4 `fp8_ds_mla` layout;
- `--hf-overrides '{"expert_dtype":"fp8"}'`, because this Flash-Base config omits `expert_dtype` and the community V4 config otherwise defaults it to `fp4` and selects an unsupported MXFP4 MoE backend.

## Baseline Launcher

The new launcher intentionally starts with a small workload:

- `--max-model-len 4096`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 4096`
- no reasoning parser
- no tool-call parser
- no MTP-specific option

The dtype defaults to `auto` so the first probe follows the model configuration. Override it explicitly when testing a Kunlun kernel requirement:

```bash
VLLM_DTYPE=bfloat16 /mnt/cluster/wangtianyu/shell/service_deepseek_v4_4gpu.sh
```

The launcher writes to `server_deepseek_v4_4gpu.log` by default. Set `LOG_PATH` to keep separate logs for separate probes.

## Adaptation Plan

1. Freeze a reproducible four-card baseline and capture the first failure.
2. Register a dedicated V4 model entry point and construct the model skeleton.
3. Inventory and load V4 weights, including FP8 UE8M0 scales, mHC, and MTP weights.
4. Validate basic layers and MoE independently of sparse attention.
5. Validate Lightning Indexer prefill, then decode and KV cache.
6. Enable end-to-end service behavior, batching, long context, MTP, and performance options one at a time.

Each step must record the commit, exact command, environment, result, failure category, and next checkpoint. Numerical failures should also save fixed inputs and intermediate tensors.

## Important V4 Differences

The checkpoint includes V4-specific fields that must not be assumed to match V3.2:

- `index_n_heads=64`
- `index_head_dim=128`
- `index_topk=512`
- compression ratios `0`, `4`, and `128`
- SWA, compressor, and Lightning Indexer caches
- compressed RoPE with `compress_rope_theta=160000`
- mHC residual mixing and final HC head
- hash routing in the first layers
- FP8 `e4m3` weights with `ue8m0` block scales

Community Intel XPU implementation details must not be copied into the Kunlun backend. Kunlun should use its existing custom ops and attention backends where the semantics match.

## Available Kunlun V4 Ops

The current container already exports several V4-specific operations from `kunlun_ops`:

- `flash_compress_4_prefill`
- `flash_compress_4_decode`
- `fused_dpsk_v4_hc_head_nofc`
- `c4a_mqa_logits`
- `c4a_paged_mqa_logits`

The MQA wrappers are implemented in `kunlun_ops/_deepgemm.py` and dispatch into the compiled `xpu_kunlun_ops` extension. The implementation plan should therefore wire the V4 model structure to these existing operations and validate their tensor contracts before considering new kernels.

## Operator and Interconnect Ledger

Every bring-up increment must update this ledger. `executed` means the operator has run in the current probe; `selected` means only backend selection or weight preparation has completed.

| Phase | Operator or operation | Implementation | Status | Collective / interconnect | Notes |
|---|---|---|---|---|---|
| platform init | Kunlun platform config | `vllm_kunlun.platforms.kunlun` | executed | none | Reads backend config and sets cache defaults |
| linear execution | FP8 block Linear | `KunlunFp8BlockScaledMMKernel` | blocked | none observed | Reached first attention projection; vLLM native input quantization fails on Kunlun `float8_e4m3fn` cast |
| linear probe | Kunlun `quant2d` | `kunlun_ops.quant2d` | selected | none | Produces per-row int8 scales; insufficient for V4 `128x128` E4M3 block scales |
| linear probe | Kunlun `cutlass_scaled_mm` | `torch.ops._C` wrapper | blocked | none | Underlying `torch.ops.xspeedgate_ops.cutlass_scaled_mm` is not registered |
| linear probe | Kunlun `matmul` | `torch.ops._C.matmul` wrapper | blocked | none | Compiled path reports `matmul unimplemented` for E4M3 tensors |
| V4 mHC | `hc_pre_kunlun_impl`, `mhc_post_fusion` | `kunlun_ops` / `xpu_kunlun_ops` | executed | none | Replaced community TileLang pre/post helpers; four-card dummy forward crossed mHC |
| linear construction | FP8 block Linear | `KunlunFp8BlockScaledMMKernel` | selected | none | Uses upstream Cutlass apply path with Kunlun support override |
| cache setup | `fp8_ds_mla` KV cache | community V4 attention config | selected | none | Requires `--kv-cache-dtype fp8` |
| MoE weight postprocess | FP8 MoE backend selection | vLLM FP8 oracle | blocked | none | `Fp8MoeBackend.NONE`; no MoE kernel has run |
| V4 sparse attention | `c4a_mqa_logits`, `c4a_paged_mqa_logits` | `kunlun_ops` / `xpu_kunlun_ops` | not reached | unknown | Must record prefill/decode separately |
| V4 compressed attention | `flash_compress_4_prefill`, `flash_compress_4_decode` | `kunlun_ops` / `xpu_flash_ops` | not reached | unknown | Must record cache layout and rank behavior |
| V4 mHC pre | `mhc_pre_tilelang` | community V4 TileLang path | blocked before execution | none observed | Import fails because `tilelang` is `None`; Kunlun exports `mhc_pre_weighted_sum`, `mhc_split_sinkhorn`, and `mhc_post_fusion` for the native replacement |
| V4 head | `fused_dpsk_v4_hc_head_nofc` | `kunlun_ops` / `xpu_kunlun_ops` | not reached | unknown | No collective expected inside the op; verify |
| MoE route | `moe_softmax_topk_norm`, `moe_sigmoid_group_topk_norm` | `torch.ops._C` | not reached | none | Router happens before expert dispatch |
| MoE dispatch | `gen_block_statistic`, `moe_pre_sorted`/`moe_pre_small` | `torch.ops._C` / `torch.ops.xspeedgate_ops` | not reached | unknown | EP path may add communication |
| MoE compute | two `moe_fc`, `silu_and_mul`, `moe_post` | `torch.ops._C` | not reached | unknown | Existing unquantized Kunlun path; FP8 path still needed |
| TP projections | column/row parallel collectives | vLLM distributed communicator | not reached | `all_reduce`/`all_gather` to verify | Current TP=4, EP is disabled |

For every executed collective, record communicator, world size, rank, tensor shape, dtype, operation order, and whether it uses XCCL/BKCL. A failure involving only one rank must include that rank's preceding operator and the last successful collective.

## Current Weight-Load Boundary

All 46 checkpoint shards (274.44 GiB) loaded successfully. The original failure occurred during FP8 MoE weight post-processing:

```text
ValueError: Unsupported FP8 MoE backend: NONE
```

The first follow-up increment adds `KunlunFp8MoEMethod`, which preserves the standard V4 FP8 weight registration but intentionally skips vLLM backend conversion. The first probe with this method completed all 46 shards on all four ranks and then reached post-load communication-buffer preparation. It exposed one local method-contract issue (`experts_cls=None` in the inherited `is_monolithic` property), not an interconnect failure. The method now declares itself monolithic so initialization can proceed to the intentionally blocked dummy forward.

The second probe completed model loading on every rank (`68.14 GiB` reported on TP0) and entered the dummy forward. The community TileLang mHC path failed during import:

```text
mhc_pre_tilelang
AttributeError: 'NoneType' object has no attribute 'PassConfigKey'
```

The Kunlun replacement now routes V4 model bindings through `hc_pre_kunlun_impl` and `mhc_post_fusion`. Small-tensor probes returned finite outputs, and the four-card probe crossed mHC successfully. The next failure is at the first attention projection FP8 Linear:

```text
RuntimeError: [NOT IMPLEMENTED]: error code=4
.../torch_xmlir/csrc/aten_capture/eager_customized/copy_kernel.cpp:407
```

The stack is in vLLM native FP8 input quantization, specifically the conversion to `torch.float8_e4m3fn`. The checkpoint uses E4M3 weights with `128x128` block scales; Kunlun `quant2d` only provides per-row int8 quantization and cannot be substituted without changing numerical semantics. A direct E4M3 `cutlass_scaled_mm` probe found that the wrapper's underlying `xspeedgate_ops.cutlass_scaled_mm` is not registered, while generic Kunlun `matmul` reports `matmul unimplemented` for E4M3 tensors.

This is classified as `kernel execution` and requires either a real Kunlun block-FP8 GEMM implementation or an explicitly scoped dequant-to-BF16 correctness fallback. No model `all_reduce` or `all_gather` was issued. XCCL/BKCL libraries loaded during setup, but no interconnect failure is indicated.

## Failure Categories

Use one category when recording a failure:

- registration
- config
- weight loading
- shape or TP
- custom-op dispatch
- kernel execution
- numerical mismatch
- KV cache
- scheduler
- API

## Basic-Layer Dummy-Forward Progress

The correctness bring-up currently uses a block-FP8 Linear fallback that transfers one loaded FP8 weight tensor to CPU, dequantizes it to BF16 with its `128x128` block scales, and transfers that layer back for `F.linear`. This path is intentionally correctness-only: it proves tensor layout, scale direction, and V4 control flow, but it is not a production GEMM implementation and does not cache a BF16 copy of the full model.

Probe V7 executed Kunlun mHC pre/post, the first attention block-FP8 Linear, and `q_a_layernorm` through `torch.ops._C.rmsnorm`. It then stopped in the community Triton `fused_q_kv_rmsnorm` kernel with `CUDA_ERROR_NOT_SUPPORTED`.

Probe V8 (`/tmp/server_deepseek_v4_fp8_bf16_fallback-v8.log`) replaced that fused kernel with two calls to `torch.ops._C.rmsnorm`. A focused probe with q shape `[7,1536]` and kv shape `[7,192]` preserved shape and BF16 dtype, produced finite outputs, and matched the CPU FP32 reference rounded to BF16 with maximum absolute error `0.0` for both tensors. The real four-rank dummy forward also crossed this operator.

The next blocker is local V4 output-projection preprocessing:

```text
attention.py:402 -> o_proj.py:48
fused_inv_rope_fp8_quant(...)
-> torch.ops.vllm.fused_inv_rope_fp8_quant_kernel
-> community Triton _fused_inv_rope_fp8_quant_per_head
-> CUDA_ERROR_NOT_SUPPORTED
```

This operator applies inverse RoPE and produces block-scaled FP8 activation plus scales for `fp8_einsum`. The next increment must adapt the output-projection pair as one semantic unit; replacing only the first Triton kernel would immediately reach the unsupported NVIDIA `fp8_einsum` path.

## Operator and Collective Ledger

| Step | Operator/path | V8 status | Communication |
|---|---|---|---|
| Model load | 46 checkpoint shards, FP8 MoE loading method | Executed on TP0-TP3 | No model collective recorded |
| Residual mixing | Kunlun mHC pre/post | Executed | Rank-local |
| Attention q_a projection | block-FP8 Linear via CPU dequant + BF16 `F.linear` | Executed, correctness-only | Rank-local replicated projection |
| q_a normalization | `torch.ops._C.rmsnorm` | Executed | Rank-local |
| Fused q/kv normalization | two Kunlun `_C.rmsnorm` calls | Executed | Rank-local |
| Attention/output preprocessing | inverse RoPE + FP8 quant Triton kernel | Blocked with `CUDA_ERROR_NOT_SUPPORTED` | Rank-local failure |
| TP all-reduce | model collective | Not reached/observed | None |
| TP all-gather | model collective | Not reached/observed | None |
| TP reduce-scatter | model collective | Not reached/observed | None |
| Runtime communicator | XCCL using `libbkcl.so`, world size 4 | Initialized on four ranks | Initialization only |

The V8 failure is therefore classified as `local custom-op/kernel dispatch`, not an XCCL/BKCL or card-interconnect failure.

## V9-V13 Layer Progress

V9 was an invalid process-lifetime probe: the launcher ran in a child shell, so the outer `wait` did not own the vLLM process and the workers received SIGTERM when `kubectl exec` disconnected. Starting with V9b, probes source the launcher in the current shell and wait for the actual server process.

V9b exposed a 20-versus-19 argument mismatch in the old `flash_mla_sparse_prefill` wrapper. A concurrent shared-workspace update replaced that call with `sparse_prefill_fwd_opt`; it was preserved rather than overwritten. V10 then showed that this kernel requires BF16 buffers for `max_logits` and `lse`. The wrapper now allocates BF16 kernel buffers and converts the returned statistics to FP32 to preserve its public contract.

V11 crossed sparse prefill and entered the Kunlun correctness-only output projection. Checkpoint metadata established the real `wo_a` layout as FP8 `[8192,4096]` with scale `[64,32]`: output groups are flattened into the first dimension and scales cover `128x128` blocks. The fallback now dequantizes this layout, reshapes it to grouped weights, executes BF16 batched matmul, sums groups, and calls the existing `wo_b` route. A scaled synthetic probe produced finite BF16 output and had maximum absolute difference `0.5` versus an FP32 reference.

V12 crossed the real inverse-RoPE/output-projection pair and stopped at post-attention RMSNorm. V13 patched RMSNorm immediately after every `DeepseekV4DecoderLayer` construction, crossed post-attention normalization, and reached the MoE router boundary. The initial `moe_ffn_block` message came from Hydrax Linear receiving an mHC list at the router gate; no expert kernel had executed yet.

Updated execution ledger:

| Step | Operator/path | V13 status | Communication |
|---|---|---|---|
| Sparse attention prefill | `torch.ops._C.sparse_prefill_fwd_opt` | Executed | Rank-local |
| Inverse RoPE | BF16 PyTorch fallback | Executed, correctness-only | Rank-local |
| `wo_a` projection | block-FP8 CPU dequant + grouped BF16 matmul | Executed, correctness-only | Rank-local |
| `wo_b` projection | existing quantized Linear route | Executed | No model collective observed |
| Post-attention norm | Kunlun `_C.rmsnorm` via decoder-instance patch | Executed | Rank-local |
| MoE router/FusedMoE boundary | mHC tuple/list -> gate -> `FusedMoE.forward_impl` | Router crossed in V14; expert input contract still blocked | No model collective observed |
| TP all-reduce/all-gather/reduce-scatter | model collective | Not reached/observed | None |
| XCCL/BKCL | world size 4, `libbkcl.so` | Initialized | Initialization only |

The V13 failure is classified as `local mHC/MoE wrapper contract`, not an interconnect failure.

## V14-V15 mHC/MoE Boundary

V14 changed only the router input from the complete mHC state list to its first item. This crossed the gate Linear and reached `FusedMoE.forward_impl`, which then asserted that expert input must be a Tensor rather than the full mHC list. V15 changed expert input to the same normalized router Tensor. One V15 run stopped earlier with transient `xpu_wait ERROR: 3` in `hc_pre_kunlun_impl`; V15b crossed that point and showed that the first item is itself a tuple.

Inspection found a shared-state consistency problem:

- The active `DeepseekV4DecoderLayer` and `DeepseekV4MoE` classes report code filenames under `vllm_kunlun/models/deepseek_v4.py`, but that source file and module spec are absent from both the repository and active site-packages.
- Their function globals belong to `vllm_kunlun.__init__`, indicating dynamic class replacement.
- The current on-disk Kunlun `mhc.py` returns a 2-item list from `mhc_pre_tilelang`, while its fused wrapper attempts a 3-item unpack.
- The canonical vLLM TileLang contract returns `(layer_input, post_layer_mix, comb_res_mix)`.
- Reloaded stale function objects are still bound at runtime, so on-disk source and executed mHC semantics are not currently identical.

The next safe increment is to make the dynamic V4 adapter source durable and unify all three mHC wrappers against the canonical 3-tensor contract before changing FusedMoE or lower-level operators. V15b still contains no model `all_reduce`, `all_gather`, or `reduce_scatter`; XCCL/BKCL remains initialized only.

## Open Baseline Questions

- Does Kunlun require `float16` or `bfloat16` compute for this checkpoint's FP8 kernels?
- Does the current vLLM 0.25.1/Kunlun combination contain the required V4 model registration and renderer support?
- Which V4 layers and caches can be represented by the existing Kunlun MLA backend?
- Should MTP be deferred until base-model generation is correct?

## Import Hook Determinism

The missing `vllm_kunlun/models/deepseek_v4.py` source path was not produced by a current import loader. A fresh module import resolves `DeepseekV4*` classes from the community NVIDIA model file, while earlier workers retained class and method objects created when the deleted Kunlun adapter file still existed. Their code objects therefore reported a stale filename and stale mHC return semantics.

The current Kunlun mHC implementation matches the active NVIDIA contract: `mhc_pre_tilelang` returns `(post_mix, comb_mix, layer_input)`, and the fused helper returns `(residual, post_mix, comb_mix, layer_input)`. The post-import hook was also made quiet during partial module initialization; validation now shows both V4 class initializers patched, mHC functions bound to `vllm_kunlun.ops.mhc`, no hook errors, and no model weights loaded. This is a registration/import fix, not a forward or interconnect result.

## V16-V20 Basic-Layer Progress

V16 loaded the 46-shard checkpoint and reached the correctness-only output projection, but the fallback summed the group dimension before `wo_b`. The upstream contract flattens `[tokens, n_groups, o_lora_rank]` to `[tokens, n_groups * o_lora_rank]`; the fallback now permutes group-major batched matmul output back to token-major and flattens it. It also returns `wo_b(projected)` directly rather than indexing `[0]`, preserving the token dimension. Focused probes confirm the corrected 2D shapes and finite output.

V17-V19 then exposed and diagnosed the mHC post wrapper boundary. `post_mix` can be `[tokens, hc_mult]` in the standalone path, so flattening it using `shape[-2]` was incorrect. The wrapper now derives `hc_mult` from residual and normalizes all four inputs explicitly. A diagnostic also confirmed that the earlier 1D `x=(4096,)` was caused by the output-projection `[0]` indexing, not by mHC layout.

V20 crossed mHC post/pre and `wo_b`, reached the actual FP8 MoE expert path on all four ranks, and stopped at the intentional guard:

```text
NotImplementedError: Kunlun FP8 MoE forward requires a validated block-scale kernel
```

This is classified as `kernel execution`. XCCL loaded `libbkcl.so`, but the log contains no model `all_reduce`, `all_gather`, or `reduce_scatter`; no interconnect failure has been observed. The next increment must validate a Kunlun block-scaled FP8 expert kernel or add a scoped correctness-only dequantized expert fallback before rerunning the full model.

## V21-V28 FP8 MoE and Indexer Progress

V21 reached the vLLM 0.25.1 modular MoE initialization guard. The correctness fallback now returns `None` from `maybe_make_prepare_finalize`, because it does not provide a modular kernel. V22 then reached hash routing and found the unavailable `_moe_C.topk_softplus_sqrt`; the Kunlun hook selects vLLM's `_topk_softplus_sqrt_torch` implementation. V23 corrected the helper/custom-op argument adapter, and a focused routing test confirmed hash-table expert IDs, `sqrt(softplus)` weights, renormalization, and routed scaling factor `2.0`.

V24 established the exact routed-expert call contract. vLLM 0.25.1 passes both `shared_experts` and `shared_experts_input` to `Fp8MoEMethod.apply`. On Kunlun the fallback has no modular kernel, so `SharedExperts` selects `NO_OVERLAP` and the runner executes it before the routed experts. The fallback now accepts these two explicit arguments and does not execute the shared expert a second time. A focused test returned finite BF16 output with shape `(4, 128)` while routing only to experts 0 and 2.

V25 loaded all 46 shards and crossed the shared-expert contract, then exposed a device-boundary bug: selected expert weights were dequantized on CPU but left there before `F.linear` consumed Kunlun activations. The fallback now copies each dequantized BF16 `w13` and `w2` back to its checkpoint parameter device for execution. A real Kunlun-device probe confirmed BF16 output and expert weights on `cuda:0`. The fallback intentionally does not retain BF16 expert matrices after use, because an unbounded per-layer cache could eventually duplicate the full FP8 expert footprint in device memory.

V26 crossed the FP8 routed expert fallback and stopped in the next attention compressor at:

```text
attention.py:389 compressor_kv_score
    torch.mm(hidden_states, fused_wkv_wgate.weight.T, out_dtype=torch.float32)
NotImplementedError: aten::mm.dtype is unavailable
```

The Kunlun plugin now registers the missing `aten::mm.dtype` CUDA overload when V4 attention loads. Its correctness fallback promotes both operands to the requested output dtype before `torch.mm`, preserving FP32 accumulation for this compressor call without replacing the attention module's `torch` object. Numerical comparison against a reference remains required.

V27 crossed that compressor GEMM and reached the Lightning Indexer Q projection/RoPE/quantization path. The new first failure is:

```text
attention.py:788 wq_b_and_q_quant
  -> common/ops/fused_indexer_q.py:427 fused_indexer_q_rope_quant
  -> Triton _fused_indexer_q_rope_quant_kernel
CompilationError: ValueError("type fp8e4nv not supported")
```

This is classified as `kernel execution / Lightning Indexer`, not communication. XCCL/BKCL initializes all four ranks in V25-V28, but no model-level `all_reduce`, `all_gather`, or `reduce_scatter` failure is present.

A cleanup review then removed the unbounded BF16 expert cache, shared the block-FP8 dequantization implementation with the linear fallback, and replaced the attention-module `torch` proxy with a CUDA backend registration for `aten::mm.dtype`. V28 verified the reviewed implementation in spawned workers: the API process and all four TP workers registered the overload, all 46 shards loaded, and execution reached the same `fused_indexer_q_rope_quant` `fp8e4nv` boundary. The next increment should adapt the fused Indexer Q/RoPE/FP8 quantization semantic unit using an existing Kunlun operator if its layout matches; otherwise add a focused correctness implementation and validate Q values, inverse/forward RoPE convention, block scales, and quantized layout before another full-model probe.
