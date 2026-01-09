# vLLM server performance

## vLLM benchmark CLI

You can directly use vLLM's CLI benchmark. For more details, please refer to[vLLM Developer Guide Benchmark Suites](https://docs.vllm.ai/en/stable/contributing/benchmarks.html)

### 1.Online testing

#### 1.1Start the vLLM server

Server startup script reference

```bash
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8000 \
      --model /xxxx/xxxx/mkdel\
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name modelname \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention", 
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2", 
                                                "vllm.mamba_mixer", 
                                                "vllm.short_conv", 
                                                "vllm.linear_attention", 
                                                "vllm.plamo2_mamba_mixer", 
                                                "vllm.gdn_attention", 
                                                "vllm.sparse_attn_indexer"]}' \

```

#### 1.2Execute test

To run the test script, you can refer to the code below.

```bash
#!/bin/bash
# Run benchmark tests
python -m vllm.entrypoints.cli.main bench serve \
    --host 127.0.0.1 \
    --port xxxx \
    --backend vllm \
    --model modelname \
    --dataset-name random \
    --num-prompts 500 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --tokenizer /xxxx/xxxx/model \
    --ignore-eos 2>&1 | tee benchmark.log
```

#### 1.3Result

The following content will be displayed after the process is complete.

```bash
========== Serving Benchmark Result ==========
Successful requests:                          500
Benchmark duration (s):                       144.89
Total input tokens:                           510414
Total generated tokens:                       512000
Request throughput (req/s):                   3.45
Output token throughput (tok/s):              3533.68
Total Token throughput (tok/s):               7056.42
----------Time to First Token----------
Mean TTFT (ms):                               57959.61
Median TTFT (ms):                             43551.93
P99 TTFT (ms):                                116202.52
----------Time per Output Token (excl. 1st token)----------
Mean TPOT (ms):                               33.30
Median TPOT (ms):                             34.15
P99 TPOT (ms):                                35.59
----------Inter-token Latency----------
Mean ITL (ms):                                33.30
Median ITL (ms):                              29.05
P99 ITL (ms):                                 46.14
============================================
```

Key Parameter Explanation:

| index                        | meaning                 | Optimization Objective   |
| --------------------------- | ------------------------------------| ---------- |
| ***\*Output Throughput\**** | Output token generation rate                   | ↑ The higher the better |
| ***\*Mean TTFT\****         | First Token Delay (Time To First Token)         | ↓ The lower the better |
| ***\*P99 TTFT\****          | 99% of requests have delayed first token.       | ↓ The lower the better |
| ***\*Mean TPOT\****         | Average generation time per output token | ↓ The lower the better |
| ***\*P99 TPOT\****          | 99% of requests' time per token generation    | ↓ The lower the better |
| ***\*ITL\****               | Delay between adjacent output tokens            | ↓ The lower the better |

### 2.Offline testing

Comming soon...

## EvalScope

EvalScope is a comprehensive model testing tool that can test not only model accuracy but also performance. For more information, please visit [website address missing].[EvalScope](https://evalscope.readthedocs.io/en/latest/index.html)，A brief introduction follows.

### 1.Download and install

EvalScope supports use in Python environments. Users can install EvalScope via pip or from source code. Here are examples of both installation methods:

```bash
#pip
pip install evalscope[perf] -U
#git
git clone https://github.com/modelscope/evalscope.git
cd evalscope
pip install -e '.[perf]'
```

After downloading, some modules may be missing, causing the program to fail to run. Just follow the prompts to install them.

### 2.Start using

The following demonstrates the performance test of the Qwen3-8B in a single-card scenario.

#### 2.1Start the server

The first step is to start the server. The example script is shown below.

```bash
python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8000 \
      --model /models/Qwen3-8B\
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 1 \
      --dtype float16 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen3-8B-Instruct \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention", 
                                                "vllm.unified_attention_with_output",
                                                "vllm.unified_attention_with_output_kunlun",
                                                "vllm.mamba_mixer2", 
                                                "vllm.mamba_mixer", 
                                                "vllm.short_conv", 
                                                "vllm.linear_attention", 
                                                "vllm.plamo2_mamba_mixer", 
                                                "vllm.gdn_attention", 
                                                "vllm.sparse_attn_indexer"]}' \

```

#### 2.2 Start EvalScope

Start EvalScope to begin performance testing.

```bash
evalscope perf \
  --parallel 1 10\#The number of concurrent requests can be tested at once, separated by spaces.
  --number 10 20\#The total number of requests per request, aligned with spaces and the concurrency count.
  --model Qwen3-8B \
  --url http://127.0.0.1:xxxx/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --tokenizer-path /xxxx/xxxx/Qwen3-8B\
  --extra-args '{"ignore_eos": true}'
```

#### 2.3Results Analysis

The following figure shows the results. You can view other data from a single test through the logs. For the specific meaning of the parameters, please refer to the parameter interpretation in the vLLM benchmark test.

```bash
Performance Test Summary Report

Basic Information:
+-------------------+------------------------+
| Model             | Qwen3-8B               |
| Total Generated   | 30,720.0 tokens        |
| Total Test Time   | 199.79 seconds         |
| Avg Output Rate   | 153.76 tokens/sec      |
+-------------------+------------------------+

Detailed Performance Metrics
+-------+------+------------+------------+-----------+-----------+-----------+-----------+-----------+---------------+
| Conc. | RPS  | Avg Lat.(s)| P99 Lat.(s)| Gen. Toks/s| Avg TTFT(s)| P99 TTFT(s)| Avg TPOT(s)| P99 TPOT(s)| Success Rate  |
+-------+------+------------+------------+-----------+-----------+-----------+-----------+-----------+---------------+
| 1     | 0.07 | 16.191     | 16.475     | 70.40      | 0.080     | 0.085     | 0.016     | 0.016     | 100.0%        |
| 10    | 0.53 | 18.927     | 19.461     | 540.87     | 0.503     | 0.562     | 0.018     | 0.019     | 100.0%        |
+-------+------+------------+------------+-----------+-----------+-----------+-----------+-----------+---------------+

Best Performance Configuration
Highest RPS:      Concurrency 10 (0.53 req/sec)
Lowest Latency:   Concurrency 1 (16.191 seconds)

Performance Recommendations:
* The system seems not to have reached its performance bottleneck, try higher concurrency
```
