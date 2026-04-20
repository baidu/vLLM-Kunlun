# FAQs

## Version Specific FAQs

- [[v0.15.1] FAQ & Feedback]

## General FAQs

### 1. What devices are currently supported?

Currently, **ONLY** Kunlun3 series(P800) series are supported

Below series are NOT supported yet:

- Kunlun4 series(M100 and M300)
- Kunlun2 series(R200)
- Kunlun1 series

We will support the kunlun4 M100 platform in early 2026.

### 2. How to get our docker containers?

Use the public Docker Hub image:

```bash
docker pull wjie520/vllm_kunlun:uv_base
```

To check whether the tag exists before downloading the full image, run:

```bash
docker manifest inspect wjie520/vllm_kunlun:uv_base >/dev/null
```

If the manifest check succeeds but `docker pull` times out, the tag exists and
the failure is usually caused by Docker Hub connectivity, proxy, or registry
mirror configuration. If the manifest check also times out, configure your
Docker network access first and retry the check. The internal registry image
listed in the installation guide is only available from the Baidu intranet.

### 3. How vllm-kunlun work with vLLM?

vllm-kunlun is a hardware plugin for vLLM. Basically, the version of vllm-kunlun is the same as the version of vllm. For example, if you use vllm 0.15.1, you should use vllm-kunlun 0.15.1 as well. For main branch, we will make sure `vllm-kunlun` and `vllm` are compatible by each commit.

### 4. How to handle the out-of-memory issue?

OOM errors typically occur when the model exceeds the memory capacity of a single XPU. For general guidance, you can refer to [vLLM OOM troubleshooting documentation](https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#out-of-memory).

In scenarios where XPUs have limited high bandwidth memory (HBM) capacity, dynamic memory allocation/deallocation during inference can exacerbate memory fragmentation, leading to OOM. To address this:

- **Limit `--max-model-len`**:  It can save the HBM usage for kv cache initialization step.

- **Adjust `--gpu-memory-utilization`**: If unspecified, the default value is `0.9`. You can decrease this value to reserve more memory to reduce fragmentation risks. See details in: [vLLM - Inference and Serving - Engine Arguments](https://docs.vllm.ai/en/latest/serving/engine_args.html#vllm.engine.arg_utils-_engine_args_parser-cacheconfig).
