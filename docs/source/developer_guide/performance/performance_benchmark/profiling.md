# Profiling



## 🔧 Action Plan（Three Phases）
### Phase 1️⃣: Multi-Device Log Redirection Configuration
#### Background
By default, kernel logs from all 8 XPU devices are interleaved and emitted to [stdout], resulting in:
- It becomes impossible to distinguish which log originates from which device.
- Timestamps become interleaved, making it difficult to analyze the temporal relationships.
- Single-device bottlenecks are masked by global aggregation.

#### Solution
During model initialization, create separate log files for each device.
#### Code Explanation (embedded in qwen2.py)
```python
import os  # ← Ensure this is imported at the top of the file
from vllm.distributed import get_tensor_model_parallel_rank  # ← Import function to get the tensor model parallel rank

class Qwen2Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer):
        super().__init__()

        # ========== [Expert Solution] Kunlun XPU Multi-Device Log Redirection ==========
        try:
            # Step 1: Get the current XPU device's rank (0~7)
            rank = get_tensor_model_parallel_rank()
            
            # Step 2: Create log directory (works with your get_kernel_time_ex.py)
            log_dir = "./xpu_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Step 3: Generate a separate log file for each device
            log_file = os.path.join(log_dir, f"rank_{rank}.log")
            
            # Step 4: Core operation – redirect file descriptors
            # os.O_TRUNC: Clear previous logs on each run to avoid mixing outputs
            fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
            os.dup2(fd, 1)  # Redirect stdout → rank_X.log
            os.dup2(fd, 2)  # Redirect stderr → rank_X.log
            os.close(fd)     # Close original file descriptor; redirection persists
            
            # Optional: print a confirmation message (will go into rank_X.log)
            print(f"[Qwen2Model Init] Rank {rank} log redirected to {log_file}")
            
        except Exception as e:
            # Fallback mechanism: failure to redirect logs does not affect model loading
            print(f"[WARNING] Failed to redirect log for rank: {e}", flush=True)
        # ========== End of log redirection code ==========

```
#### ⚠️ Common Issues
**Q1**:Why not use Python's `logging` module?
**A**:The XPU runtime kernel logs are emitted from the C++ layer and cannot be captured by Python’s `logging` module. Redirection via low-level file descriptors is required.
**Q1**:Will logs be lost if the model fails to load??
**A**:The `try-except` block ensures that if log redirection fails, it falls back to the default behavior without affecting model startup.

### Phase 2️⃣: Profiling Environment Activation
#### 🚀 vLLM Launch
```bash
unset XPU_DUMMY_EVENT
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPU_USE_MOE_SORTED_THRES=1
export XFT_USE_FAST_SWIGLU=1
export XMLIR_CUDNN_ENABLED=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export VLLM_HOST_IP=$(hostname -i)
echo "VLLM_HOST_IP: $VLLM_HOST_IP"

export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false

export XPUAPI_DEBUG=0x1              # Enable kernel performance logging
export XPURT_DISPATCH_MODE=PROFILING # Activate profiling mode

USE_ORI_ROPE=1 VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port 8000 \
      --model /models/Qwen2.5-72B-Instruct \
      --gpu-memory-utilization 0.9 \
      --trust-remote-code \
      --max-model-len 32768 \
      --tensor-parallel-size 8 \
      --dtype float16 \
      --max_num_seqs 512 \
      --max_num_batched_tokens 32768 \
      --max-seq-len-to-capture 32768 \
      --block-size 128 \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --distributed-executor-backend mp \
      --served-model-name Qwen2.5-72B-Instruct \
      --compilation-config '{"splitting_ops": ["vllm.unified_attention_with_output_kunlun",
            "vllm.unified_attention", "vllm.unified_attention_with_output",
            "vllm.mamba_mixer2"]}' 2>&1 | tee output_p800.log

```


#### 🚀 Client Load Testing
```bash
#!/bin/bash

# Define test combinations array (concurrency x input length x output length)
TEST_COMBINATIONS=(
    "8x1024x1024" # Medium-low concurrency
)

# Create result directory
RESULT_DIR="bench_$(date +%Y%m%d_%H%M)"
mkdir -p $RESULT_DIR

# Summary results file
SUMMARY_FILE="$RESULT_DIR/summary_results.csv"
echo "num_prompts,input_len,output_len,throughput,latency_mean,latency_p50,latency_p90,latency_p99" >$SUMMARY_FILE

# Progress counter
TOTAL_TESTS=${#TEST_COMBINATIONS[@]}
CURRENT_TEST=0

# Loop through different test combinations
for COMBINATION in "${TEST_COMBINATIONS[@]}"; do
    # Parse combination parameters
    NUM_PROMPTS=$(echo $COMBINATION | cut -d'x' -f1)
    INPUT_LEN=$(echo $COMBINATION | cut -d'x' -f2)
    OUTPUT_LEN=$(echo $COMBINATION | cut -d'x' -f3)

    # Update progress
    CURRENT_TEST=$((CURRENT_TEST + 1))

    echo "=========================================================="
    echo "Test progress: $CURRENT_TEST/$TOTAL_TESTS ($(printf "%.1f" $(echo "$CURRENT_TEST/$TOTAL_TESTS*100" | bc -l))%)"
    echo "Current test configuration: concurrency=$NUM_PROMPTS, input length=$INPUT_LEN, output length=$OUTPUT_LEN"
    echo "=========================================================="

    OUTPUT_FILE="$RESULT_DIR/p800_${NUM_PROMPTS}_${INPUT_LEN}_${OUTPUT_LEN}.log"

    # Run benchmark
    python3 -m vllm.entrypoints.cli.main bench serve \
        --host 127.0.0.1 \
        --port 8000 \
        --backend vllm \
        --model Qwen2.5-72B-Instruct \
        --dataset-name random \
        --num-prompts $NUM_PROMPTS \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --tokenizer /ssd1/models/Qwen2.5-72B-Instruct \
        --ignore-eos 2>&1 | tee $OUTPUT_FILE

    # Wait 15 seconds to let the service recover
    echo "Waiting 15 seconds before the next round..."
    sleep 15

    # Extract key performance metrics from output and append to summary file
    THROUGHPUT=$(grep "Throughput" $OUTPUT_FILE | awk '{print $2}')
    LATENCY_MEAN=$(grep "Mean latency" $OUTPUT_FILE | awk '{print $3}')
    LATENCY_P50=$(grep "p50 latency" $OUTPUT_FILE | awk '{print $3}')
    LATENCY_P90=$(grep "p90 latency" $OUTPUT_FILE | awk '{print $3}')
    LATENCY_P99=$(grep "p99 latency" $OUTPUT_FILE | awk '{print $3}')

    echo "$NUM_PROMPTS,$INPUT_LEN,$OUTPUT_LEN,$THROUGHPUT,$LATENCY_MEAN,$LATENCY_P50,$LATENCY_P90,$LATENCY_P99" >>$SUMMARY_FILE
done

# Output summary report
echo "=========================================================="
echo "Benchmark completed! Results saved in: $RESULT_DIR"
echo "=========================================================="


```

### Phase 3️⃣: Log Analysis and Bottleneck Identification
```text
xpu_logs/
├─ rank_0.log
├─ rank_1.log
├─ rank_2.log
├─ rank_3.log
├─ rank_4.log
├─ rank_5.log
├─ rank_6.log
└─ rank_7.log

```
#### 🔍 Script Workflow (op_log.py)
**Input**:Raw Kernel Logs (Sample Format)
```
[XPURT_PROF] void xblas_xpu3::fc_cdnn_infer<float16,...> 123456 ns
[XPURT_PROF] void kl3_all_reduce<float16> 987654 ns
```
**Processing logic**
:::::{tab-set}
::::{tab-item} op_log.py 


```python
"""
A better version of 'get_op_time.py', get more level dump and support kl3.
 
Usage: python3 get_kernel_time_ex.py --help
"""
 
import os
import sys
import re
 
unit_factors = [0.9, 1.3, 1.45] # kunlun1, kunlun2, kunlun3
patterns = ["\[XPURT_PROF\] (\S+)\s+\S+\s+(\S+) ns", "\[XPURT_PROF\] (\S+)\s+(\S+)\s+\S+ ns"]
tab_space_num = int(4)
 
def get_total_time(res):
    total_time = 0.0
    for i in res.values():
        total_time += i
    return  total_time
 
def print_info_op(res, cnt, unit, op):
    total_time = get_total_time(res)
    total_cnt = 0
    # print detailed op time
    lis=sorted(res.items(), key=lambda d:d[1], reverse=True)
    if sys.version_info.major == 2:
        import commands
        for i in range(len(lis)):
            (status, cmd_output) = commands.getstatusoutput("c++filt {}".format(lis[i][0]))
            if status == 0:
                formt_type = (cmd_output.split('('))[0]
            total_cnt += cnt[lis[i][0]]
    elif sys.version_info.major == 3:
        import subprocess
        for i in range(len(lis)):
            (status, cmd_output) = subprocess.getstatusoutput("c++filt {}".format(lis[i][0]))
            if status == 0:
                formt_type = (cmd_output.split('('))[0]
            total_cnt += cnt[lis[i][0]]
    print(f"{op} {total_time / unit} {total_cnt}")
 
def print_info_kernel(res, cnt, unit):
    total_time = get_total_time(res)
    total_cnt = 0
    print("Total time(ms) is {}".format(total_time / unit))
    # print detailed op time
    lis=sorted(res.items(), key=lambda d:d[1], reverse=True)
    if sys.version_info.major == 2:
        print("{:<90}{:<10}{:<15}{:<15}".format("Op type", "count", "time(ms)", "%"))
        import commands
        for i in range(len(lis)):
            (status, cmd_output) = commands.getstatusoutput("c++filt {}".format(lis[i][0]))
            if status == 0:
                formt_type = (cmd_output.split('('))[0]
            print("{:<90}{:<10}{:<15}{:<15.5}".format(formt_type, cnt[lis[i][0]], lis[i][1] / unit, \
                lis[i][1] / total_time * 100))
            total_cnt += cnt[lis[i][0]]
    elif sys.version_info.major == 3:
        print("{:<90}{:<10}{:<20}{:<20}".format("Op type", "count", "time(ms)", "%"))
        import subprocess
        for i in range(len(lis)):
            (status, cmd_output) = subprocess.getstatusoutput("c++filt {}".format(lis[i][0]))
            if status == 0:
                formt_type = (cmd_output.split('('))[0]
            print("{:<150}{:<10}{:<25}{:<20.5}".format(formt_type, cnt[lis[i][0]], lis[i][1] / unit, \
                lis[i][1] / total_time * 100))
            total_cnt += cnt[lis[i][0]]
 
    print("Total count is {}".format(total_cnt))
 
def count_head_spaces(s: str) -> int:
   
    count = 0
    for char in s:
        if char == ' ':
            count += 1
        else:
            break
    return count
 
def process_line(lines, pattern1, unit_factor, dump_level):
    """ process a line in a file with profiling info
 
    Args:
        unit_factor: A factor differentiated by KUNLUN1 and KUNLUN2
 
    """
    res = {}
    cnt = {}
    op = "init_op"
    unit = unit_factor * 1000 * 1000 # ns -> ms
    wait_next_one = False
    for i in range(len(lines)):
        cur_line = lines[i]
        if "gtest_" in cur_line:
            cur_level = count_head_spaces(cur_line) / tab_space_num
            if cur_level == dump_level:
                wait_next_one = False
                print_info_op(res, cnt, unit, op)
                # clear buf
                res = {}
                cnt = {}
                op = cur_line.lstrip().rstrip()
            elif cur_level < dump_level:
                wait_next_one = True
                # skip record kernel time untime next one
                continue
        if wait_next_one:
            # skip record kernel time
            continue
        match = re.match(pattern1, lines[i])
        if match:
            op_type = match.group(1)
            op_time = match.group(2)
            if op_type in res:
                res[op_type] += float(op_time)
                cnt[op_type] += 1
            else:
                res[op_type] = float(op_time)
                cnt[op_type] = 1
 
    # get left total time
    if dump_level == -1:
        print_info_kernel(res, cnt, unit)
    else:
        print_info_op(res, cnt, unit, op)
    return res
 
def process_file(file_name, pattern2, unit_factor, dump_level = -1):
    """ Process a file line by line
 
    Iteratively process each line in the target file.
 
    """
 
    with open(file_name, "r") as f:
        lines = f.readlines()
        f1_res_list = process_line(lines, pattern2, unit_factor, dump_level)
 
if __name__ == '__main__':
    import argparse
 

    parser = argparse.ArgumentParser()
 

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-xpu1', action='store_true', help='指定为 xpu1')
    group.add_argument('-xpu2', action='store_true', help='指定为 xpu2')
    group.add_argument('-xpu3', action='store_true', help='指定为 xpu3')
    parser.add_argument('--level', type=int, default=-1, help='指定 dump 缩进级别（默认为 -1）')

    parser.add_argument('filename', help='要处理的文件名')
 

    args = parser.parse_args()
 

    filename = args.filename
    xpu_version = 0
    if args.xpu2:
        xpu_version = 1
    if args.xpu3:
        xpu_version = 2
    dump_level = args.level
    print(f'Filename: {filename}')
    print(f'-xpu option: {xpu_version}')
    print(f'--level option: {dump_level}')
 
    unit_factor = unit_factors[xpu_version]
    pattern_idx = 0
    if xpu_version > 0:
        pattern_idx = 1
    process_file(filename, patterns[pattern_idx], unit_factor, dump_level)
 
```

::::

::::{tab-item} op_log.sh



```bash

for i in {0..7}; do
    python op_log.py -xpu3 xpu_logs/rank_${i}.log > analysis_rank${i}.log
    echo "Rank ${i} 分析完成"
done


for i in {0..7}; do
    echo "=== Rank $i ===" 
    head -n 6 analysis_rank${i}.log | tail -n 5
done
```
::::
:::::
#### 📈 Output Example (analysis_rank0.log)
```
Filename: xpu_logs/rank_0.log
-xpu option: 2
--level option: -1
Total time(ms) is 53742.29571862069
Op type                                                                                   count     time(ms)            %                   
void xblas_xpu3::fc_cdnn_infer<float16, float16, float16, float16, float, float, float, float, 1>                                                     661569    22736.262780689656       42.306              
void kl3_all_reduce<float16>                                                                                                                          176134    14782.525712413793       27.506              
void kl3_all_reduce_butterfly<float16>                                                                                                                164864    4197.28395862069         7.81           
```
#### 🚨 Troubleshooting Guide
|Symptom|Cause|Solution|
|-|-|-|
|`xpu_logs` directory is empty|XPUAPI_DEBUG not enabled|Verify that the environment variable is correctly set|
All 8 log files have identical content|Multi-process backend not activated|Ensure `--distributed-executor-backend` mp is specified|
|Throughput drops >15%|Profiling overhead too high|Enable profiling only during analysis; disable in production|
