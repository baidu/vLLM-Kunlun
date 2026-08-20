# Unified attention_benchmarks

This directory is the single source used for both vLLM-Kunlun v0.11.0 and v0.15.1 tests.

Version-specific behavior is handled by runtime capability detection in `runner.py`, `mla_runner.py`, and `common.py`.

Set `ATTN_BENCH_MODEL_DIR` when running standard attention if the local Llama config is not under `models/meta-llama/Meta-Llama-3-8B` next to this directory.

Active configs:

- `configs/standard_attention.yaml` — standard attention with `KUNLUN_ATTN`
- `configs/mla_decode.yaml` — MLA decode with `KUNLUN_FLASHMLA`
- `configs/mla_prefill.yaml` — MLA prefill with `KUNLUN_FLASHMLA`
- `configs/mla_mixed_batch.yaml` — MLA chunked prefill + decode mixed batches with `KUNLUN_FLASHMLA`

Run commands:

```bash
python benchmark.py --config configs/standard_attention.yaml --output-json standard_attention.json --output-csv standard_attention.csv
python benchmark.py --config configs/mla_decode.yaml --output-json mla_decode.json --output-csv mla_decode.csv
python benchmark.py --config configs/mla_prefill.yaml --output-json mla_prefill.json --output-csv mla_prefill.csv
python benchmark.py --config configs/mla_mixed_batch.yaml --output-json mla_mixed_batch.json --output-csv mla_mixed_batch.csv
```

Validation status on P800 133:

- Standard attention passed smoke coverage for prefill, decode, mixed,
  speculative decode, and context extension shapes on both v0.11.0 and v0.15.1.
- MLA pure decode and pure prefill passed smoke coverage on both v0.11.0 and
  v0.15.1.
- MLA chunked prefill / context extension requires the benchmark compatibility
  layer to route `merge_attn_states` to Kunlun's `attention_merge_stage`.
