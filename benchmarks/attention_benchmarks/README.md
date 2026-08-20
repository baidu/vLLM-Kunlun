# vLLM-Kunlun Attention Benchmarking Suite

Fast, flexible benchmarking for vLLM attention and MLA backends with an extended batch specification grammar. This is the Kunlun port of the upstream `attention_benchmarks` suite — the structure mirrors upstream and only the backend selection / hardware-specific bits are adapted for Kunlun.

## Quick Start

```bash
cd benchmarks/attention_benchmarks

# Run a pre-configured benchmark
python benchmark.py --config configs/standard_attention.yaml
python benchmark.py --config configs/mla_decode.yaml
python benchmark.py --config configs/mla_prefill.yaml
python benchmark.py --config configs/mla_mixed_batch.yaml

# Or run custom benchmarks
python benchmark.py \
    --backends KUNLUN_ATTN \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --output-csv results.csv
```

## Simplified Batch Specification Grammar

Express workloads concisely using query length and sequence length:

```python
"q2k"              # 2048-token prefill (q_len=2048, seq_len=2048)
"q1s1k"            # Decode: 1 token with 1K sequence
"8q1s1k"           # 8 decode requests
"q4s1k"            # 4-token extend (e.g., spec decode)
"2q2k_32q1s1k"     # Mixed: 2 prefills + 32 decodes
"16q4s1k"          # 16 spec decode (4 tokens each)
```

### Grammar Rule

```text
Format: (<count>?) q<q_len>(k?) (s<seq_len>(k?))?

- count:   Number of identical requests (optional, default=1)
- q_len:   Query length (number of new tokens)
- seq_len: Total sequence length (optional, defaults to q_len for prefill)
- 'k':     Multiplies value by 1024

Mixed batches: Use _ to combine (e.g., "2q2k_32q1s1k")
```

**Note**: Decode, prefill, and spec decode are just different query lengths - no special syntax needed!

## Pre-configured Benchmarks

The suite includes several pre-configured YAML benchmark configurations:

### Standard Attention Benchmark

Tests Kunlun standard attention with pure prefill, decode, mixed batches, speculative decode, and context extension shapes.

```bash
python benchmark.py --config configs/standard_attention.yaml
```

### MLA Decode Benchmark

Tests pure decode performance on the Kunlun MLA backend with varying batch sizes and sequence lengths.

```bash
python benchmark.py --config configs/mla_decode.yaml
```

### MLA Prefill Benchmark

Tests Kunlun MLA pure prefill, chunked prefill, and context extension shapes.

```bash
python benchmark.py --config configs/mla_prefill.yaml
```

### MLA Mixed Batch Benchmark

Tests Kunlun MLA chunked prefill performance with mixed prefill + decode batches.

```bash
python benchmark.py --config configs/mla_mixed_batch.yaml
```

---

## Universal Benchmark

The `benchmark.py` script handles **all** Kunlun backends - both standard attention and MLA.

### Standard Attention (KUNLUN_ATTN)

```bash
python benchmark.py \
    --backends KUNLUN_ATTN \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --num-layers 10 \
    --repeats 5 \
    --output-csv results.csv
```

### MLA Backends

```bash
python benchmark.py \
    --backends KUNLUN_FLASHMLA \
    --batch-specs "64q1s1k" "64q1s4k" \
    --output-csv mla_results.csv
```

### Parameter Sweeps

Use `--sweep-param` and `--sweep-values` to run parameter sweeps from the CLI:

#### Reorder Batch Threshold Optimization

**Question:** What's the optimal `reorder_batch_threshold` for speculative decoding?

```bash
python benchmark.py \
    --backend KUNLUN_FLASHMLA \
    --batch-specs "q4s1k" "q8s2k" \
    --sweep-param reorder_batch_threshold \
    --sweep-values 1 4 16 64 256 512 \
    --output-csv threshold_sweep.csv
```

### All Command-Line Options

```text
--config CONFIG                     # Path to YAML config file (overrides other args)
--backends BACKEND [BACKEND ...]    # KUNLUN_ATTN, KUNLUN_FLASHMLA, KUNLUN_FLASHMLA_SPARSE
--backend BACKEND                   # Single backend (alternative to --backends)
--batch-specs SPEC [SPEC ...]       # Batch specifications using extended grammar

# Model configuration
--num-layers N                      # Number of layers
--head-dim N                        # Head dimension
--num-q-heads N                     # Query heads
--num-kv-heads N                    # KV heads
--block-size N                      # Block size

# Benchmark settings
--device DEVICE                     # Device (default: cuda:0)
--repeats N                         # Repetitions
--warmup-iters N                    # Warmup iterations
--profile-memory                    # Profile memory usage
--kv-cache-dtype {auto,fp8}
--cuda-graphs / --no-cuda-graphs

# Parameter sweeps
--sweep-param PARAM                 # Parameter name to sweep (e.g., reorder_batch_threshold)
--sweep-values N [N ...]            # Values to sweep for the parameter

# Output
--output-csv FILE                   # Save to CSV
--output-json FILE                  # Save to JSON
```

`--prefill-backends` and `num_kv_splits` are retained only for upstream
argument/config compatibility. The Kunlun-only MLA runner rejects non-empty
prefill backend selection and rejects `num_kv_splits`.

## Hardware Requirements

| Backend | Hardware |
| ------- | -------- |
| KUNLUN_ATTN | Kunlun P800+ |
| KUNLUN_FLASHMLA | Kunlun P800+ |
| KUNLUN_FLASHMLA_SPARSE | Kunlun P800+ |

`standard_attention.yaml` uses a local Llama 3 config directory. By default the
runner reads `benchmarks/attention_benchmarks/models/meta-llama/Meta-Llama-3-8B`;
override it with `ATTN_BENCH_MODEL_DIR` when needed.

## Using MLA Runner Directly

All Kunlun MLA backends are available through `mla_runner.run_mla_benchmark()`:

```python
from mla_runner import run_mla_benchmark
from common import BenchmarkConfig

config = BenchmarkConfig(
    backend="KUNLUN_FLASHMLA",
    batch_spec="64q1s4k",
    num_layers=10,
    head_dim=576,
    num_q_heads=128,
    num_kv_heads=1,
    block_size=64,
    device="cuda:0",
    repeats=5,
    warmup_iters=3,
)

result = run_mla_benchmark("KUNLUN_FLASHMLA", config, reorder_batch_threshold=64)
print(f"Time: {result.mean_time:.6f}s")
```

## Python API

```python
from batch_spec import parse_batch_spec, format_batch_spec, get_batch_stats
from common import BenchmarkConfig, BenchmarkResult, ResultsFormatter

# Parse batch specs
requests = parse_batch_spec("2q2k_q4s1k_32q1s1k")
print(format_batch_spec(requests))
# "2 prefill (2x2k), 1 extend (1xq4kv1k), 32 decode (32x1k)"

# Get batch statistics
stats = get_batch_stats(requests)
print(f"Total tokens: {stats['total_tokens']}")
print(f"Num decode: {stats['num_decode']}, Num prefill: {stats['num_prefill']}")

# Format results
formatter = ResultsFormatter()
formatter.save_csv(results, "output.csv")
formatter.save_json(results, "output.json")
```

## Tips

**1. Warmup matters** - Use `--warmup-iters 10` for stable results

**2. Multiple repeats** - Use `--repeats 20` for low variance

**3. Save results** - Always use `--output-csv` or `--output-json`

**4. Test incrementally** - Start with `--num-layers 1 --repeats 1`

**5. Extended grammar** - Leverage spec decode, chunked prefill patterns

**6. Parameter sweeps** - Use `--sweep-param` and `--sweep-values` to find optimal values
