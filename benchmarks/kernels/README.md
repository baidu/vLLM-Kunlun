# MoE Kernel Benchmark

`benchmark_moe.py` is based on the vLLM MoE kernel benchmark and keeps the
upstream benchmark flow: model config loading, shape derivation, Ray worker
dispatch, CUDA graph capture, and per-batch latency reporting.

Kunlun dispatch is auto-detected: if the running PyTorch is the Kunlun build
(`torch_xmlir` is importable) the script imports
`vllm_kunlun.vllm_utils_wrapper` and dispatches to
`vllm_kunlun.ops._kunlun_ops.KunlunOps`:

- non-EP: `KunlunOps.fused_moe`
- EP: `KunlunOps.fused_moe_ep`

The EP wrapper supports both observed `fused_moe_ep` signatures so the same
script can run on v0.11.0 and v0.15.1 style containers. On non-Kunlun
PyTorch the upstream Triton + CUDA Graph path is used unchanged.

On v0.15.1-style Kunlun containers, the benchmark initializes the vLLM v1
workspace manager before running `KunlunOps.fused_moe`. This is required for
non-EP large-batch shapes that use the shared workspace path inside
`fused_moe`; v0.11.0 containers that do not expose `vllm.v1.worker.workspace`
fall back to a no-op.

## Smoke Configuration

The default smoke values are recorded in `configs/qwen3_30b_a3b_smoke.env`.
The default `MODEL` points to the config-only Qwen3-30B-A3B directory under
`models/Qwen3-30B-A3B`. The benchmark only needs `config.json` for shape
derivation; no model weights are loaded.

```bash
source configs/qwen3_30b_a3b_smoke.env
python benchmark_moe.py --model "$MODEL" --tp-size "$TP_SIZE" --batch-size $BATCH_SIZES
```

For EP:

```bash
python benchmark_moe.py --model "$MODEL" --tp-size "$TP_SIZE" --enable-expert-parallel --batch-size $BATCH_SIZES
```

For a full batch sweep:

```bash
source configs/qwen3_30b_a3b_smoke.env
python benchmark_moe.py --model "$MODEL" --tp-size "$TP_SIZE" --batch-size $FULL_BATCH_SIZES
python benchmark_moe.py --model "$MODEL" --tp-size "$TP_SIZE" --enable-expert-parallel --batch-size $FULL_BATCH_SIZES
```

## Limitations

- The Kunlun path currently covers unquantized fp16/bf16 fused MoE
  (see `KUNLUN_SUPPORTED_DTYPES` in `benchmark_moe.py`; extend it once more
  dtypes are validated).
- `--dtype fp8_w8a8`, `--dtype int8_w8a16`, and `--use-deep-gemm` are not
  supported by the Kunlun path and will raise `NotImplementedError`.
- `--tp-size` in EP mode is a microbenchmark expert partition parameter, not a
  full multi-rank serving launch.
