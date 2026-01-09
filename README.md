![vLLM Kunlun Logo](vllm_kunlun/patches/vLLM_Kunlun.jpg)

<p align="center">
  <a href="https://vllm-kunlun.readthedocs.io/en/latest/"><b>  Documentation</b></a> |
  <a href="https://vllm-kunlun.readthedocs.io/en/latest/quick_start.html"><b>  Quick Start</b></a> |
  <a href="https://join.slack.com/t/vllm-kunlun/shared_invite/zt-3iinb8u5z-FcqZKbNNdMJ_32fHmipzvw"><b>  Slack</b></a>
</p>

---

## Latest News 🔥
- [2025/12] Initial release of vLLM Kunlun

---

# Overview

vLLM Kunlun (vllm-kunlun) is a community-maintained hardware plugin designed to seamlessly run vLLM on the Kunlun XPU. It is the recommended approach for integrating the Kunlun backend within the vLLM community, adhering to the principles outlined in the [RFC Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162). This plugin provides a hardware-pluggable interface that decouples the integration of the Kunlun XPU with vLLM.

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

<h3>Generaltive Models</h3>
<table>
  <thead>
    <tr>
      <th width="30%">Model</th>
      <th width="12%">Support</th>
      <th width="15%">Quantization</th>
      <th width="10%">LoRA</th>
      <th width="20%">Piecewise Kunlun Graph</th>
      <th width="23%">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">Qwen2</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen2.5</td>
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
      <td class="model-name">Qwen3-Moe</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3-Next</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">MiMo-V2-Flash</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Llama2</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Llama3</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Llama3.1</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">gpt-oss</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">DeepSeek-R1</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">DeepSeek-V3</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">DeepSeek-V3.2</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
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
      <td class="model-name">Qwen3-VL</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
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
| v0.11.0 | Latest stable version | [QuickStart](https://vllm-kunlun.readthedocs.io/en/latest/quick_start.html) and [Installation](https://vllm-kunlun.readthedocs.io/en/latest/installation.html) for more details |

---

## Contribute to vLLM Kunlun

If you're interested in contributing to this project, please read [Contributing](CONTRIBUTING.md) to vLLM Kunlun.

## Star History 🔥

We opened the project at Dec 8, 2025. We love open source and collaboration ❤️

[![Star History Chart](https://api.star-history.com/svg?repos=baidu/vLLM-Kunlun&type=date&legend=bottom-right)](https://www.star-history.com/#baidu/vLLM-Kunlun&type=date&legend=bottom-right)

## Sponsors 👋

We sincerely appreciate the [**KunLunXin**](https://www.kunlunxin.com/) team for their support in providing XPU resources, which enabled efficient model adaptation debugging, comprehensive end-to-end testing, and broader model compatibility.

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
