# Operator accuracy test

## torch_xray

torch_xray is an operator precision analysis tool that can dump module-level input-output precision comparisons and automatically construct operator unit tests.

### 1.Download and install

***\*python3.10:\****

bos:/klx-sdk-release-public/xpytorch/dev_kl3/torch_xray/latest/torch_xray-999.9.9-cp310-cp310-linux_x86_64.whl

﻿[https://su.bcebos.com/klx-sdk-release-public/xpytorch/dev_kl3/torch_xray/latest/](https://su.bcebos.com/klx-sdk-release-public/xpytorch/dev_kl3/torch_xray/latest/torch_xray-999.9.9-py3-none-any.whl)torch_xray-999.9.9-cp310-cp310-linux_x86_64.whl

***\*python3.8:\****

bos:/klx-sdk-release-public/xpytorch/dev_kl3/torch_xray/latest/torch_xray-999.9.9-cp38-cp38-linux_x86_64.whl

﻿[https://su.bcebos.com/klx-sdk-release-public/xpytorch/dev_kl3/torch_xray/latest/](https://su.bcebos.com/klx-sdk-release-public/xpytorch/dev_kl3/torch_xray/latest/torch_xray-999.9.9-py3-none-any.whl)torch_xray-999.9.9-cp38-cp38-linux_x86_64.whl

Note that the same installation package must be used when using it in different environments.

### 2.Use

#### Dump module-level inputs and outputs and compare their precision.

Below is a sample code snippet used to dump the input and output of the vision module and compare the errors in the vllm framework.

```bash
from torch_xray import PrecisionDebugger

def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    # dump_path # Path to store dump results
    # rank # Rank that needs to be dumped
    # step # Setting the inference value to 1 is sufficient.
    # model # The module to be dumped must be of type nn.module
        debugger = PrecisionDebugger(dump_path="dump-vision", hook_name="dump", rank=[0], step=[1], model=self.model.visual, dump_torch_api=False)
        debugger.start()
        ........
```

The results directory will generate an h5 file and a csv file.

```bash
-rw-r--r-- 1 root root 471231309 Oct 31 13:12 globalrank-0_localrank-0.h5
-rw-r--r-- 1 root root        71 Oct 31 13:11 globalrank-0_localrank-0_summary.csv
```

#### Data processing

```bash
summary xxx.h5 sum.txt
```

The generated h5 file is processed using the summary command to generate a txt file in which the results are presented in tabular form.

```bash
+-------+------+------+-----------------------------------------------------------+-------------+-------------+--------------+-------------+
| Index | Step | Rank | Module                                                    |         Min |         Max |         Mean |         Std |
+-------+------+------+-----------------------------------------------------------+-------------+-------------+--------------+-------------+
|     0 |    1 |    0 | patch_embed.proj.Conv3d.0.forward_params.weight           | -0.0776367  | 0.0795898   |      6.8e-06 | 0.0072608   |
|     1 |    1 |    0 | patch_embed.proj.Conv3d.0.forward_params.bias             | -3.046875   | 2.953125    |    0.0113748 | 0.3257138   |
|     2 |    1 |    0 | patch_embed.proj.Conv3d.0.forward_input.0                 | -0.7490234  | 0.7021484   |    0.3302804 | 0.2339017   |
|     3 |    1 |    0 | patch_embed.proj.Conv3d.0.forward_output.0                | -4.0078125  | 5.1210938   |    0.0147052 | 0.3815643   |
|     4 |    1 |    0 | pos_embed.Embedding.0.forward_params.weight               | -13.8125    | 20.25       |    0.0010043 | 0.2428094   |
|     5 |    1 |    0 | pos_embed.Embedding.0.forward_input.0                     |        0.0  | 2303.0      | 1153.9191895 | 714.594360  |
|     6 |    1 |    0 | pos_embed.Embedding.0.forward_output.0                    | -13.8125    | 20.25       |    0.0007552 | 0.2643428   |
|     7 |    1 |    0 | rotary_pos_emb.Qwen2_5_VisionRotaryEmbedding.0.forward... |        0.0  | 25.0        |    1.7337022 | 3.9271674   |
|     8 |    1 |    0 | blocks.0.norm1.LayerNorm.0.forward_params.weight          | -0.5351562  | 3.140625    |    0.4660275 | 0.7907906   |
|     9 |    1 |    0 | blocks.0.norm1.LayerNorm.0.forward_params.bias            | -2.359375   | 2.921875    |    0.0013793 | 0.1879374   |
|    10 |    1 |    0 | blocks.0.norm1.LayerNorm.0.forward_input.0                | -15.65625   | 20.21875    |    0.0155256 | 0.4382802   |
|    11 |    1 |    0 | blocks.0.norm1.LayerNorm.0.forward_output.0               | -6.1640625  | 6.7460938   |    0.0006746 | 0.2708515   |
|    12 |    1 |    0 | blocks.0.attn.qkv.QKVParallelLinear.0.forward_params.bias | -6.125      | 6.1875      |   -0.0292423 | 0.8602651   |
|    13 |    1 |    0 | blocks.0.attn.qkv.QKVParallelLinear.0.forward_input.0     | -6.1640625  | 6.7460938   |    0.0006746 | 0.2708515   |
|    14 |    1 |    0 | blocks.0.attn.qkv.QKVParallelLinear.0.forward_output.0    | -6.5859375  | 7.6171875   |   -0.0125549 | 1.0678084   |
|    15 |    1 |    0 | blocks.0.attn.proj.RowParallelLinear.0.forward_params...  | -3.578125   | 3.203125    |   -0.0043617 | 0.4846557   |
|    16 |    1 |    0 | blocks.0.attn.proj.RowParallelLinear.0.forward_input.0    | -1.9130859  | 1.4375      |    0.0005577 | 0.0947055   |
|    17 |    1 |    0 | blocks.0.attn.proj.RowParallelLinear.0.forward_output.0   | -9.109375   | 7.3867188   |   -0.0034284 | 0.4465481   |
|    18 |    1 |    0 | blocks.0.norm2.LayerNorm.1.forward_params.weight          | -0.1376953  | 14.5625     |    1.9166113 | 3.017405    |
|    19 |    1 |    0 | blocks.0.norm2.LayerNorm.1.forward_params.bias            | -1.6328125  | 3.84375     |    0.0062865 | 0.2443586   |
|    20 |    1 |    0 | blocks.0.norm2.LayerNorm.1.forward_input.0                | -8.5859375  | 11.109375   |    0.0120974 | 0.4243064   |
|    21 |    1 |    0 | blocks.0.norm2.LayerNorm.1.forward_output.0               | -12.015625  | 14.265625   |   -0.0012364 | 0.4973041   |
|    22 |    1 |    0 | blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forwar...  | -9.4375     | 0.7304688   |   -2.4200516 | 1.6754951   |
|    23 |    1 |    0 | blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forwar...  | -12.015625  | 14.265625   |   -0.0012364 | 0.4973041   |
|    24 |    1 |    0 | blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forwar...  | -12.59375   | 13.0625     |   -2.1465943 | 1.8433502   |
|    25 |    1 |    0 | blocks.0.mlp.act_fn.GELU.0.forward_input.0                | -12.59375   | 13.0625     |   -2.1465943 | 1.8433502   |
+-------+------+------+-----------------------------------------------------------+-------------+-------------+--------------+-------------+
```

#### Accuracy Comparison

```bash
# The results are stored in result.csv
compare xpu.h5 gpu.h5 result.csv
```

The `compare` command is used to process the H5 files generated on the GPU and XPU, resulting in a CSV file. This CSV file is then downloaded to the local machine and opened with Excel, yielding a result similar to the image below.

If you encounter a "no matched keys" problem, please refer to the instructions at the end of this article for a solution.


#### Example of results

```bash
+-------+--------+-----------------------------------------------------------+--------+-----------+-------------+-------------+--------+
| Index | Status | Module (Bench/Target)                                     | Cosine |      RMSE | IsClose (%) | Max Err (t) |  GtNum |
+-------+--------+-----------------------------------------------------------+--------+-----------+-------------+-------------+--------+
|     0 |        | patch_embed.proj.Conv3d.0.forward_params.weight           |      1 |         0 |         100 |           0 |      0 |
|     1 |        | patch_embed.proj.Conv3d.0.forward_params.bias             |      1 |         0 |         100 |           0 |      0 |
|     2 |        | patch_embed.proj.Conv3d.0.forward_input.0                 |      1 |         0 |         100 |           0 |      0 |
|     3 |        | patch_embed.proj.Conv3d.0.forward_output.0                |      1 |  9.90E-06 |         100 |    0.001953 |    267 |
|     4 |        | pos_embed.Embedding.0.forward_params.weight               |      1 |         0 |         100 |           0 |      0 |
|     5 |        | pos_embed.Embedding.0.forward_input.0                     |      1 |         0 |         100 |           0 |      0 |
|     6 |        | pos_embed.Embedding.0.forward_output.0                    |      1 |         0 |         100 |           0 |      0 |
|     7 |        | rotary_pos_emb.Qwen2_5_VisionRotaryEmbedding.0.forward... |      1 |         0 |         100 |           0 |      0 |
|     8 |        | blocks.0.norm1.LayerNorm.0.forward_params.weight          |      1 |         0 |         100 |           0 |      0 |
|     9 |        | blocks.0.norm1.LayerNorm.0.forward_params.bias            |      1 |         0 |         100 |           0 |      0 |
|    10 |        | blocks.0.norm1.LayerNorm.0.forward_input.0                |      1 |  1.14E-05 |         100 |  0.00390625 |    216 |
|    11 |        | blocks.0.norm1.LayerNorm.0.forward_output.0               |      1 |  1.84E-05 |       99.98 |   0.0078125 |   1585 |
|    12 |        | blocks.0.attn.qkv.QKVParallelLinear.0.forward_params.bias |      1 |         0 |         100 |           0 |      0 |
|    13 |        | blocks.0.attn.qkv.QKVParallelLinear.0.forward_input.0     |      1 |  1.84E-05 |       99.98 |   0.0078125 |   1585 |
|    14 |        | blocks.0.attn.qkv.QKVParallelLinear.0.forward_output.0    |      1 | 0.0002776 |       99.53 |  0.00390625 | 119074 |
|    15 |        | blocks.0.attn.proj.RowParallelLinear.0.forward_params...  |      1 |         0 |         100 |           0 |      0 |
|    16 |        | blocks.0.attn.proj.RowParallelLinear.0.forward_input.0    |      1 |  3.40E-05 |       99.07 |   0.0012207 |  52482 |
|    17 |        | blocks.0.attn.proj.RowParallelLinear.0.forward_output.0   |      1 | 0.0001283 |       99.07 |  0.00390625 |  50591 |
|    18 |        | blocks.0.norm2.LayerNorm.1.forward_params.weight          |      1 |         0 |         100 |           0 |      0 |
|    19 |        | blocks.0.norm2.LayerNorm.1.forward_params.bias            |      1 |         0 |         100 |           0 |      0 |
|    20 |        | blocks.0.norm2.LayerNorm.1.forward_input.0                |      1 | 0.0001437 |       99.01 |   0.0039062 |  31376 |
|    21 |   Fail | blocks.0.norm2.LayerNorm.1.forward_output.0               |      1 | 0.0002779 |       98.72 |    0.015625 |  40770 |
|    22 |        | blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forward... |      1 |         0 |         100 |           0 |      0 |
|    23 |   Fail | blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forward... |      1 | 0.0002779 |       98.72 |    0.015625 |  40770 |
|    24 |        | blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forward... |      1 | 0.000779  |       98.67 |   0.0078125 | 196313 |
|    25 |        | blocks.0.mlp.act_fn.GELU.0.forward_input.0                |      1 | 0.000779  |       98.67 |   0.0078125 | 196313 |
|    26 |        | blocks.0.mlp.act_fn.GELU.0.forward_output.0               |      1 | 0.0001012 |       98.08 |   0.0039062 | 153508 |
+-------+--------+-----------------------------------------------------------+--------+-----------+-------------+-------------+--------+
```

Generally, the main focus is on Min Err/Max Err.

#### Indicator Explanation

To be improved...

### The dump operator is tested and run.

```bash
X_DEBUG=0x102 # trace operator name、arguments shape、dtype、data_range
X_DEDUP=True # Remove duplicates based on shape and dtype. 
X_DUMP_NUM # The default value is 0, meaning no tensor data is saved. Setting it to n means that n parameters are randomly selected from each operator to save the actual parameters.
```

Below is a sample code snippet that dumps information such as the size and dtype of the forward operator of Qwen3_VisionTransformer. During runtime, an xray_debug directory will be automatically created in the current directory to store the dump results.

```bash
from torch_xray import begin_dump, end_dump
.............
﻿
class Qwen3_VisionTransformer(nn.Module):
﻿
    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        ..........
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        # Start dump 
        # X_DEBUG=0x102 # trace operator name、arguments shape、dtype、data_range
        # X_DEDUP=True # Remove duplicates based on shape and dtype.
        # The default value is 0, meaning no tensor data is saved. Setting it to n means that n parameters are randomly selected from each operator to save the actual parameters.
        begin_dump(X_DEBUG=0x102, X_DEDUP=True, X_DUMP_NUM=5)
        
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)
        ...........
        
        # End dump
        end_dump(clear_context=True)
        return hidden_states
```
This is the file directory.
```bash
├── xary_debug/                
│   ├── proc_xxx/     # Process-based storage results
│       ├── dump/     # The dumped tensor
│       ├── dump.json # Information needed to generate unit tests, such as input/output size and dtype.
```

#### Generate unit test

jprof --cpu_init --blacklist --factory=load dump.json

Create a pytests directory in the current directory to store unit tests.

#### Run unit test

The GPU only needs to copy the XPU's pytests directory and execute it.

Since the unit test program defaults to finding the actual dumped tensors using relative paths, this step must be performed in the xary_debug/ directory.

```bash
# detail_compare_path stores the unit test results.
pytest --detail_compare_path=./xxx.csv proc_xxx/pytests/ --seed 42
```

#### Results Comparison

```bash
# After obtaining two result CSV files, compare them and generate result.csv.
summary_diff_check  ./xpu.csv ./gpu.csv ./result.csv
```

#### Example of results

```bash
+------------+-----------------------+-------------+-------------+-----------+----------+---------+---------+----------+
| name       | op_name               | dtype       | shape       |   min-val |  max-val | is_pass | xpu_max |  gpu_max |
+------------+-----------------------+-------------+-------------+-----------+----------+---------+---------+----------+
| 00004-aten | aten.linspace.default | torch.float | [10]        |         0 |       47 | pass    |       0 | 1.91E-06 |
| 00005-aten | aten.linspace.default | torch.float | [26]        |         0 |       47 | pass    |       0 |        0 |
| 00027-aten | aten.add.Tensor       | torch.int64 | [10, 26]    |         0 |        0 | pass    |       0 |        0 |
| 00028-aten | aten.add.Tensor       | torch.int64 | [10, 26]    |         0 |        0 | pass    |       0 |        0 |
| 00037-aten | aten.add.Tensor       | torch.float | [260, 1152] | -29.09375 |    33.75 | pass    |       0 |        0 |
| 00038-aten | aten.add.Tensor       | torch.float | [260, 1152] | -27.1875  |   37.625 | pass    |       0 |        0 |
| 00047-aten | aten.add.Tensor       | torch.float | [260, 1152] | -28.98438 | 42.34375 | pass    |       0 |        0 |
| 00082-aten | aten.sub.Tensor       | torch.int32 | [1]         |         0 |        0 | pass    |       0 |        0 |
+------------+-----------------------+-------------+-------------+-----------+----------+---------+---------+----------+
```

The main focus is on the values ​​of gpu_1e-1, xpu_1e-1, etc., which represent the number of elements whose error between the gpu/xpu result and the cpu result exceeds the order of 1e-n. This serves as the primary basis for determining whether there is a problem with the operator's precision.

### Replenish

#### Bypassing the issue of differing naming conventions between Kunlun Card and GPU modules, which prevents diff calculation.

```bash
#
blocks.0.mlp.linear_fc1.ColumnParallelLinear.0.forward_params.bias
#
blocks.0.mlp.linear_fc1.ColumnParalleLinear.forward_params.bias
```

As shown in the figure above, due to various reasons, the module names dumped by the GPU and XPU are often different, and the compare command cannot be used to identify them directly.

```python
for step in steps: # (['/'] for group creation order h5py >= 3.10.0)
    # for bench_key, target_key in get_matched_names(
    #     list(dump_ben[str(step)].keys()),
    #     list(dump_tar[str(step)].keys()),
    #     fuzzy_match,
    # ):
    for bench_key, target_key in zip(
        list(dump_ben[str(step)].keys()),
        list(dump_tar[str(step)].keys()),
):
```

Modify torch_xray/compare/compare.py to skip the get_matched_name step. This modification will allow for line-by-line comparison even if module names differ, producing a compare result. However, it's crucial to ensure that the number of rows in the GPU and XPU dumps is consistent.