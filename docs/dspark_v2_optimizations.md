# V2 GPUModelRunner DSpark (DFlash) Optimizations for vLLM 0.25.1 (Kunlun XPU)

This change carries the Kunlun-side optimizations for running DSpark (DFlash)
speculative decoding on the V2 GPUModelRunner path
(`VLLM_USE_V2_MODEL_RUNNER=1`) of upstream vLLM 0.25.1
(`__version__ = 0.25.1`, `__commit_id__ = g752a3a504`).

The files under the top-level `vllm/` directory are the final patched versions
of the corresponding upstream `vllm/v1/worker/gpu/**` files, exported from the
accuracy-verified runtime (image `dspark-exp-snapshot:20260804-fullopt`).
Apply them on top of an upstream vLLM 0.25.1 install.

## Modification points

All optimizations are gated by env switches and individually revertible:

| Optimization | File (`vllm/v1/worker/gpu/`) | Switch | Description |
|---|---|---|---|
| Host control-plane vectorization | `block_table.py`, `input_batch.py` | `KUNLUN_HOSTVEC` | Remove per-request `int(tensor)` D2H syncs; batch the host-side bookkeeping |
| Post-update vectorization | `input_batch.py` | `KUNLUN_HOSTVEC_POSTUPDATE` | `_post_update_vec`: merge host ops in the post-update phase |
| Post-update v3 | `input_batch.py` | `KUNLUN_POSTUPD3` | `_post_update_vec3`: remove 3 more D2H syncs |
| Greedy sampling fast path | `sample/sampler.py` | `KUNLUN_GUMBEL_GREEDY` | Direct argmax when temp=0 for all requests; byte-identical to gumbel greedy |
| DFlash draft-input vectorization | `spec_decode/dflash/speculator.py` | `KUNLUN_DFLASH_VEC` (default ON) | Vectorize draft model input preparation |
| Rejection sampling on XPU | `spec_decode/rejection_sampler.py` | `KUNLUN_REJ_XPU` | Run rejection sampling on XPU, removing host round-trips |

Release configuration: all six switches ON.

## Performance

Qwen3-8B + DSpark, P800 single card (TP=1), c32 concurrent throughput (tokens/s):

| Configuration | c32 throughput |
|---|---:|
| V1 runner + DSpark (reference) | 1189.23 |
| V2 runner + DSpark, partial switches (host-vec + gumbel + postupd3) | 1036.84 |
| **V2 runner + DSpark, all optimizations (release config)** | **1159.21** |

The V2-vs-V1 gap narrows from ~13% to ~2.5%.

## Accuracy (aligned)

Full ifeval (1082 prompts, max_tokens=30000, temp=0):

| Configuration | prompt_strict | inst_strict | prompt_loose | inst_loose |
|---|:-:|:-:|:-:|:-:|
| All optimizations, V2 + DSpark (release config, TP=1) | **0.8253** | 0.8771 | 0.8632 | 0.9041 |
| Pre-optimization baseline (TP=1) | 0.8253 | — | — | — |

Outputs are byte-identical to the baseline (greedy lossless): accuracy is aligned.
