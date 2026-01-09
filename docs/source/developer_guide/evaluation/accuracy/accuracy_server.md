# Overall accuracy test

## EvalScope

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

### 2.Dataset preparation script

```python
from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data
import os  # Step 1: Import the os module

schema = CollectionSchema(
    name="VL-Test",
    datasets=[
        CollectionSchema(
            name="PureText",
            weight=1,
            datasets=[
                DatasetInfo(
                    name="mmlu_pro",
                    weight=1,
                    task_type="exam",
                    tags=["en"],
                    args={"few_shot_num": 0},
                ),
                DatasetInfo(
                    name="ifeval",
                    weight=1,
                    task_type="instruction",
                    tags=["en"],
                    args={"few_shot_num": 0},
                ),
                DatasetInfo(
                    name="gsm8k",
                    weight=1,
                    task_type="math",
                    tags=["en"],
                    args={"few_shot_num": 0},
                ),
            ],
        ),
        CollectionSchema(
            name="Vision",
            weight=2,
            datasets=[
                DatasetInfo(
                    name="math_vista",
                    weight=1,
                    task_type="math",
                    tags=["en"],
                    args={"few_shot_num": 0},
                ),
                DatasetInfo(
                    name="mmmu_pro",
                    weight=1,
                    task_type="exam",
                    tags=["en"],
                    args={"few_shot_num": 0},
                ),
            ],
        ),
    ],
)


# get the mixed data
mixed_data = WeightedSampler(schema).sample(1000)

output_path = "outputs/vl_test.jsonl"  # Step 2: Define the output file path
output_dir = os.path.dirname(output_path)  # Step 3: Obtain the directory name
if not os.path.exists(output_dir):  # Step 4: Check if the directory exists
    os.makedirs(output_dir, exist_ok=True)  # Step 5: Automatically create directories


# dump the mixed data to a jsonl file
dump_jsonl_data(mixed_data, output_path)  # Step 6: Securely write to the file
```

Dataset composition visualization:

```
┌───────────────────────────────────────┐
│       VL-Test (1000 samples)          │
├─────────────────┬─────────────────────┤
│   PureText      │      Vision         │
│   (333 samples) │    (667 samples)    │
├─────────────────┼─────────────────────┤
│ • mmlu_pro      │ • math_vista        │
│ • ifeval        │ • mmmu_pro          │
│ • gsm8k         │                     │
└─────────────────┴─────────────────────┘
```

### 3.Test

```python
from dotenv import dotenv_values

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model="Qwen2.5-VL-7B-Instruct",
    api_url="http://localhost:8804/v1",
    api_key="EMPTY",
    eval_type=EvalType.SERVICE,
    datasets=[
        "data_collection",
    ],
    dataset_args={
        "data_collection": {
            "local_path": "../outputs/vl_test.jsonl",
        }
    },
    eval_batch_size=5,
    generation_config={
        "max_tokens": 30000,  # The maximum number of tokens that can be generated should be set to a large value to avoid output truncation.
        "temperature": 0.6,  # Sampling temperature (recommended value from qwen report)
        "top_p": 0.95,  # top-p sampling (recommended value from qwen report)
        "top_k": 20,  # Top-k sampling (recommended value from qwen report)
        "n": 1,  # Number of responses generated per request
        "repetition_penalty": 1.0,  # 1.0 = Penalty disabled, >1.0 = Penalty repeated.
    },
)

run_task(task_cfg=task_cfg)
```

Parameter Tuning Guide:

| Parameter         | Current value | Effect                                   | Adjustment suggestions                                   |
| ----------------- | ------------- | ---------------------------------------- | -------------------------------------------------------- |
| `temperature`     | 0.6           | Control output diversity                 | Math problems ↓ 0.3 / Creative writing ↑ 0.9             |
| `top_p`           | 0.95          | Filtering low-probability tokens         | Reduce "nonsense"                                        |
| `eval_batch_size` | 5             | Number of requests processed in parallel | With sufficient video memory, it can be increased to 10. |

Run the test:

```bash
#!/bin/bash
# ========================================
# Step 1: Set the log file path
# ========================================
LOG_FILE="accuracy_$(date +%Y%m%d_%H%M).log"

# ========================================
# Step 2: Execute the Python script and capture all output
# Meaning of 2>&1:
# - 2 represents standard error output (stderr)
# ->& represents redirection and merging
# - 1 represents standard output (stdout)
# Function: Merges error messages into standard output as well.
# ========================================
python accuracy.py 2>&1 | tee "$LOG_FILE"

# ========================================
# Step 3: Check Execution Status
# ${PIPESTATUS[0]} Get the exit code of the first command (Python) in the pipeline
# ========================================
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed! Log saved to: $LOG_FILE"
else
    echo "❌ Evaluation failed! Exit code: $EXIT_CODE Please check the log: $LOG_FILE"
fi
```

### 4.Common problem fixes

#### 4.1 NLTK resource missing fix

```bash
Resource punkt_tab not found.
```

Solution：

```python
import nltk
import os

# Step 1: Set the download path (select a writable directory)
download_dir = "/workspace/myenv/nltk_data"
os.makedirs(download_dir, exist_ok=True)

# Step 2: Configure NLTK data path
nltk.data.path.append(download_dir)

# Step 3: Download necessary resources
print("🔽 Start downloading punkt_tab resource...")
try:
    nltk.download("punkt_tab", download_dir=download_dir)
    print("✅ Download successful!")
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("💡 Alternative: Download manually from GitHub")
    print(
        "   URL: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
    )
```

repair:

```bash
# Activate environment
source /workspace/myenv/bin/activate

# Run the repair script
python fix_nltk.py

# Rerun the test
bash run_accuracy_test.sh
```

### 5.Results Display

```bash
+-------------+---------------------+--------------+---------------+-------+
|  task_type  |       metric        | dataset_name | average_score | count |
+-------------+---------------------+--------------+---------------+-------+
|    exam     |         acc         |   mmmu_pro   |     0.521     |  334  |
|    math     |         acc         |  math_vista  |    0.6066     |  333  |
|    exam     |         acc         |   mmlu_pro   |    0.5405     |  111  |
| instruction | prompt_level_strict |    ifeval    |    0.6937     |  111  |
|    math     |         acc         |    gsm8k     |    0.8288     |  111  |
+-------------+---------------------+--------------+---------------+-------+
```
