![vLLM Kunlun Logo](vllm_kunlun/patches/vLLM_Kunlun.jpg)

<p align="center">
  <a href="https://vllm-kunlun.readthedocs.io/en/latest/"><b>Documentation</b></a> |
  <a href=""><b>Users Forum</b></a> |
  <a href="https://vllm-kunlun.slack.com/"><b>slack</b></a> |
</p>

---

## Latest News🔥
- [2025/12] Initial release of vLLM Kunlun

---

# Overview

vLLM Kunlun (vllm-kunlun) is a community-maintained hardware plugin designed to seamlessly run vLLM on the Kunlun XPU. It is the recommended approach for integrating the Kunlun backend within the vLLM community, adhering to the principles outlined in the [RFC]: Hardware pluggable. This plugin provides a hardware-pluggable interface that decouples the integration of the Kunlun XPU with vLLM.

By utilizing the vLLM Kunlun plugin, popular open-source models, including Transformer-like, Mixture-of-Expert, Embedding, and Multi-modal LLMs, can run effortlessly on the Kunlun XPU.

---
## Prerequisites

- **Hardware**: Kunlun3 P800 
- **OS**: Ubuntu 22.04 
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.5.1
  - vLLM (same version as vllm-kunlun)

---
## Supported Models

<h3>Generative Models</h3>
<table>
  <thead>
    <tr>
      <th width="23%">Model</th>
      <th width="12%">Support</th>
      <th width="15%">Quantization</th>
      <th width="10%">LoRA</th>
      <th width="20%">Piecewise Kunlun Graph</th>
      <th width="20%">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">Qwen2/2.5</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3-Moe/Coder</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">QwQ-32B</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">LLama2/3/3.1</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">GLM-4.5/Air</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3-next</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">coming soon</span></td>
    </tr>
    <tr>
      <td class="model-name">GPT OSS</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">coming soon</span></td>
    </tr>
    <tr>
      <td class="model-name">DeepSeek-v3/3.2</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">coming soon</span></td>
    </tr>
  </tbody>
</table>

<h3>Multimodal Language Models</h3>
<table>
  <thead>
    <tr>
      <th width="20%">Model</th>
      <th width="12%">Support</th>
      <th width="15%">Quantization</th>
      <th width="10%">LoRA</th>
      <th width="20%">Piecewise Kunlun Graph</th>
      <th width="23%">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">Qianfan-VL</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen2.5-VL</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">InternVL2.5/3/3.5</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">InternS1</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen2.5-Omni</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">coming soon</span></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3-VL</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">coming soon</span></td>
    </tr>
  </tbody>
</table>



## Performance Visualization 🚀
### High-performance computing at work: How different models perform on the Kunlun3 P800.

Current environment: 16-way concurrency, input/output size 2048.


![Models and tgs](./vllm_kunlun/patches/performance.png)

## Getting Started

Please use the following recommended versions to get started quickly:

| Version | Release type | Doc |
|----------|---------------|-----|
| v0.10.1.1 | Latest stable version | [QuickStart](https://vllm-kunlun.readthedocs.io/en/latest/quick_start.html) and [Installation](https://vllm-kunlun.readthedocs.io/en/latest/installation.html) for more details |

---

## Contributing

See [CONTRIBUTING](https://vllm-kunlun.readthedocs.io/en/latest/developer_guide/contribution/index.html) for more details, which is a step-by-step guide to help you set up the development environment, build, and test.

We welcome and value any contributions and collaborations:
- Open an [Issue](https://github.com/baidu/vLLM-Kunlun/issues) if you find a bug or have a feature request

## License

Apache License 2.0, as found in the [LICENSE](https://github.com/baidu/vLLM-Kunlun/blob/main/LICENSE.txt) file.