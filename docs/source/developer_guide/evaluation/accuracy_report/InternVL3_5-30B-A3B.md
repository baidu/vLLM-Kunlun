# InternVL3_5-30B-A3B

* vLLM Version: vLLM: 0.10.1.1 , vLLM-KunLun Version: v0.10.1.1
* Software Environment:OS: Ubuntu 22.04, PyTorch ≥ 2.5.1
* Hardware Environment: KunLun P800
* Parallel mode:TP8

```
+-------------+---------------------+--------------+---------------+-------+
|  task_type  |       metric        | dataset_name | average_score | count |
+-------------+---------------------+--------------+---------------+-------+
|    exam     |         acc         |   mmmu_pro   |    0.5449     |  334  |
|    math     |         acc         |  math_vista  |    0.6847     |  333  |
|    exam     |         acc         |   mmlu_pro   |    0.6126     |  111  |
| instruction | prompt_level_strict |    ifeval    |    0.7658     |  111  |
|    math     |         acc         |    gsm8k     |    0.9369     |  111  |
+-------------+---------------------+--------------+---------------+-------+
```
