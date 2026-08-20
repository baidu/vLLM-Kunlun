# vLLM-Kunlun Benchmarks

Benchmark assets for vLLM-Kunlun validation, kept close to the upstream vLLM
benchmark layout.

## Contents

- `attention_benchmarks/`: vLLM attention microbenchmark harness with the
  Kunlun standard attention and Kunlun MLA backends. See
  [`attention_benchmarks/README.md`](attention_benchmarks/README.md).
- `kernels/benchmark_moe.py`: upstream-style vLLM MoE kernel benchmark with an
  opt-in Kunlun fused MoE path. See [`kernels/README.md`](kernels/README.md).

The scripts are intentionally kept close to upstream benchmark structure. The
Kunlun-specific code is limited to backend selection and operator dispatch, so
the actual measured path still enters `vllm` and `vllm_kunlun` modules.
