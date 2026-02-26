# Quantization Guide
>Note: This feature is currently experimental. In future versions, there may be behavioral changes around configuration, coverage, performance improvement.

Like vLLM, we now support quantization methods such as compressed-tensors, AWQ, and GPTQ, enabling various precision configurations including W8A8, W4A16, and W8A16. These can help reduce memory consumption and accelerate inference while preserving model accuracy.


## Support Matrix
<table border="1" style="border-collapse: collapse; width: auto; margin: 0 0 0 0; text-align: center;">
  <thead>
    <tr>
      <td colspan="2" style="padding: 10px; font-weight: bold; border: 1px solid #000;">Compressed-Tensors (w8a8-Int8)</td>
      <td colspan="4" style="padding: 10px; font-weight: bold; border: 1px solid #000;">Weight only (w4a16/w8a16)</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #000;">Dynamic</td>
      <td style="padding: 10px; border: 1px solid #000;">Static</td>
      <td colspan="1" style="padding: 10px; border: 1px solid #000;">AWQ (w4a16)</td>
      <td colspan="2" style="padding: 10px; border: 1px solid #000;">GPTQ (w4a16/w8a16)</td>
       <td colspan="1" style="padding: 10px; border: 1px solid #000;">Compressed-Tensors (w4a16)</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #000;">Dense/MoE</td>
      <td style="padding: 10px; border: 1px solid #000;">Dense/MoE</td>
      <td style="padding: 10px; border: 1px solid #000;">Dense/MoE</td>
      <td style="padding: 10px; border: 1px solid #000;">Dense</td>
      <td style="padding: 10px; border: 1px solid #000;">MoE</td>
      <td style="padding: 10px; border: 1px solid #000;">Dense/MoE</td>
    </tr>
  </thead>
  <tbody>
    <tr style="height: 40px;">
      <td style="padding: 10px; border: 1px solid #000;">✅</td>
      <td style="padding: 10px; border: 1px solid #000;">✅</td>
      <td style="padding: 10px; border: 1px solid #000;">✅</td>
      <td style="padding: 10px; border: 1px solid #000;">✅</td>
      <td style="padding: 10px; border: 1px solid #000;">WIP</td>
      <td style="padding: 10px; border: 1px solid #000;">✅</td>
    </tr>
  </tbody>
</table>

+ Compressed-Tensors w8a8-Int8 dynamic and static quantization are supported for all LLMs and VLMs.
+ Compressed-Tensors w4a16 are supported for all LLMs and VLMs.
+ AWQ(w4a16) quantization is supported for all LLMs and VLMs.
+ GPTQ (w4a16/w8a16) quantization is supported for all dense models.

## Usages

### Compressed-tensor
To run a `compressed-tensors` model with vLLM-Kunlun, you can use `Qwen/Qwen3-30B-A3B-Int8` with the following command:

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Int8 \
    --quantization compressed-tensors
```


### AWQ

To run an `AWQ` model with vLLM-Kunlun, you can use `Qwen/Qwen3-32B-AWQ` with the following command:

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --quantization awq
```


### GPTQ

To run a `GPTQ` model with vLLM-Kunlun, you can use `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` with the following command:

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --quantization gptq
```

