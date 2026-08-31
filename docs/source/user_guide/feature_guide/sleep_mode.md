# Sleep Mode Guide

## Overview

Sleep mode offloads or discards the model weights and the KV cache from device
memory while the engine stays alive, so another process can use the freed XPU
memory. It is mainly used by reinforcement learning (RL) post-training
workloads (PPO, GRPO, DPO, ...), where the same device is shared between the
vLLM rollout engine and the trainer.

The API is the upstream vLLM one, see
[vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html).
vLLM Kunlun makes it work on Kunlun XPU; no Kunlun-specific API or flag is
introduced.

Two sleep levels are supported, same semantics as upstream:

- **Level 1**: weights are offloaded to host memory, the KV cache is discarded.
  Use it when the same model will be reused after `wake_up()`. Make sure the
  host has enough free memory to hold the weights.
- **Level 2**: both weights and KV cache are discarded. Use it when the weights
  will be replaced anyway, e.g. when the trainer pushes updated weights, or
  when switching to another model.

## Usage

Offline inference:

```python
from vllm import LLM

llm = LLM(model="/models/Qwen3-8B", enable_sleep_mode=True)
llm.generate("Hello, how are you?")

llm.sleep(level=1)  # free device memory
llm.wake_up()       # restore weights and KV cache

# level 2 + weight update from the trainer
llm.sleep(level=2)
llm.wake_up(tags=["weights"])   # re-map the weights pool only
llm.collective_rpc("reload_weights")
llm.wake_up(tags=["kv_cache"])  # re-allocate the KV cache
```

Online serving:

```shell
VLLM_SERVER_DEV_MODE=1 vllm serve /models/Qwen3-8B --enable-sleep-mode
```

```shell
curl -X POST 'http://127.0.0.1:8000/sleep?level=1'
curl -X GET  'http://127.0.0.1:8000/is_sleeping'
curl -X POST 'http://127.0.0.1:8000/wake_up'
```

The `/sleep`, `/wake_up` and `/is_sleeping` endpoints are only exposed when
`VLLM_SERVER_DEV_MODE=1`. They are unauthenticated administrative endpoints, so
only enable them in a trusted development environment.

## Kunlun Implementation Notes

Sleep mode relies on `vllm.device_allocator.cumem.CuMemAllocator`, which backs
the weights and KV cache pools with CUDA VMM allocations (`cuMemCreate` +
`cuMemMap`). Two Kunlun-specific adaptations live in
`vllm_kunlun/device_allocator/cumem.py`, both installed through post-import
hooks registered in `vllm_kunlun/__init__.py`:

1. **Offload copies go through the CUDA driver API.** Upstream
   `sleep()` / `wake_up()` move the pool contents with the CUDA *runtime*
   `cudaMemcpy`, which does not accept the VMM virtual addresses on Kunlun XPU
   and crashes. During the two calls the module-global `libcudart` is replaced
   by a passthrough shim whose `cudaMemcpy` is routed to `cuMemcpyDtoH_v2` /
   `cuMemcpyHtoD_v2`. The direction is pinned per call (`sleep()` only copies
   device to host, `wake_up()` only host to device), because Kunlun runs with
   `CUDA_FAKE_UVA_ENABLE=1` where the direction cannot be inferred from the
   pointer.
2. **`reload_weights()` re-maps the weights pool when needed.** After a level-2
   sleep the weights pool is unmapped. Calling `reload_weights()` without a
   preceding `wake_up(tags=["weights"])` would write into released device memory
   and fail with XPU error `-707`. `GPUModelRunner.reload_weights` is wrapped so
   the weights pool is woken up first if it is still unmapped.

Both patches are idempotent, and are skipped when the cumem extension is not
available (in which case sleep mode is unavailable anyway).
