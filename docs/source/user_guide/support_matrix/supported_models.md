# Supported Models

For guidance on finding Hugging Face, ModelScope, private object storage,
approved internal sources, or local filesystem paths for these model families,
see {doc}`model_sources`.

## Generative Models

| Model         | Support | W8A8 | LoRA | Tensor Parallel | Expert Parallel | Data Parallel | Piecewise Kunlun Graph |
| :------------ | :------ | :--- | :--- | :-------------- | :-------------- | :------------ | :--------------------- |
| Qwen3         | ✅       | ✅    | ✅    | ✅               |                 | ✅             | ✅                      |
| Qwen3-Moe     | ✅       | ✅    | ✅    | ✅               | ✅               | ✅             | ✅                      |
| Qwen3-Next    | ✅       | ✅    | ✅    | ✅               | ✅               | ✅             | ✅                      |
| Deepseek v3.2 | ✅       | ✅    |      | ✅               |                 | ✅             | ✅                      |

## Multimodal Language Models
| Model    | Support | W8A8 | LoRA | Tensor Parallel | Expert Parallel | Data Parallel | Piecewise Kunlun Graph |
| :------- | :------ | :--- | :--- | :-------------- | :-------------- | :------------ | :--------------------- |
| Qwen3-VL | ✅       | ✅    |      | ✅               |                 | ✅             | ✅                      |
