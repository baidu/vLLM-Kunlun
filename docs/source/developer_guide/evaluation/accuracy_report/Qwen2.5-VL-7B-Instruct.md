# Qwen2.5-VL-7B-Instruct

* vLLM Version: vLLM: 0.10.1.1 , vLLM-KunLun Version: v0.10.1.1
* Software Environment:OS: Ubuntu 22.04, PyTorch ≥ 2.5.1
* Hardware Environment: KunLun P800
* Parallel mode:TP1

```
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
