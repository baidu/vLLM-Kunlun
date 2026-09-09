# Supported Models

## Generative Models

| Model         | Support | W8A8 | LoRA | Tensor Parallel | Expert Parallel | Data Parallel | Piecewise Kunlun Graph |
| :------------ | :------ | :--- | :--- | :-------------- | :-------------- | :------------ | :--------------------- |
| Qwen3         | ✅       | ✅    | ✅    | ✅               |                 | ✅             | ✅                      |
| Qwen3-Moe     | ✅       | ✅    | ✅    | ✅               | ✅               | ✅             | ✅                      |
| Qwen3-Next    | ✅       | ✅    | ✅    | ✅               | ✅               | ✅             | ✅                      |
| Qwen3.8       | ✅       | ✅    |      |                  |                 |              | ✅                      |
| Deepseek v3.2 | ✅       | ✅    |      | ✅               |                 | ✅             | ✅                      |

Qwen3.8 (27B-W8A8): verified on a single XPU (`tensor_parallel_size=1`) with a
compressed-tensors W8A8 checkpoint; multi-XPU tensor parallel is pending
verification. See the [Qwen3.8 tutorial](../../tutorials/single_xpu_Qwen3.8-27B-W8A8.md).

## Multimodal Language Models
| Model    | Support | W8A8 | LoRA | Tensor Parallel | Expert Parallel | Data Parallel | Piecewise Kunlun Graph |
| :------- | :------ | :--- | :--- | :-------------- | :-------------- | :------------ | :--------------------- |
| Qwen3-VL | ✅       | ✅    |      | ✅               |                 | ✅             | ✅                      |
