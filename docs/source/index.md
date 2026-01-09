# Welcome to vLLM Kunlun Plugin

:::{figure} ./logos/vllm-kunlun-logo-text-light.png
:align: center
:alt: vLLM
:class: no-scaled-link
:width: 70%
:::

:::{raw} html

<p style="text-align:center">
<strong>vLLM Kunlun Plugin
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/baidu/vLLM-Kunlun" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/baidu/vLLM-Kunlun/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/baidu/vLLM-Kunlun/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

vLLM Kunlun (vllm-kunlun) is a community-maintained hardware plugin designed to seamlessly run vLLM on the Kunlun XPU. It is the recommended approach for integrating the Kunlun backend within the vLLM community, adhering to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162). This plugin provides a hardware-pluggable interface that decouples the integration of the Kunlun XPU with vLLM.

By utilizing the vLLM Kunlun plugin, popular open-source models, including Transformer-like, Mixture-of-Expert, Embedding, and Multi-modal LLMs, can run effortlessly on the Kunlun XPU.

## Documentation

% How to start using vLLM on Kunlun XPU?
:::{toctree}
:caption: Getting Started
:maxdepth: 1
quick_start
installation
tutorials/index.md
faqs
:::

% What does vLLM Kunlun Plugin support?
:::{toctree}
:caption: User Guide
:maxdepth: 1
user_guide/support_matrix/index
user_guide/configuration/index
user_guide/feature_guide/index
user_guide/release_notes
:::

% How to contribute to the vLLM Kunlun project
:::{toctree}
:caption: Developer Guide
:maxdepth: 1
developer_guide/contribution/index
developer_guide/feature_guide/index
developer_guide/evaluation/index
developer_guide/performance/index
:::

% How to involve vLLM Kunlun
:::{toctree}
:caption: Community
:maxdepth: 1
community/governance
community/contributors
community/versioning_policy
community/user_stories/index
:::
