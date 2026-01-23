#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Starting vLLM server in container ${DOCKER_NAME}"

sudo docker exec -d "${DOCKER_NAME}" bash -lc "
  set -e

  chmod +x \"${GITHUB_WORKSPACE}/vLLM-Kunlun/setup_env.sh\"
  source \"${GITHUB_WORKSPACE}/vLLM-Kunlun/setup_env.sh\"

  rm -f ${VLLM_LOG}
  export XPU_VISIBLE_DEVICES=${XPU_VISIBLE_DEVICES}

  python -u -m vllm.entrypoints.openai.api_server \
    --host ${VLLM_HOST} \
    --port ${VLLM_PORT} \
    --model ${MODEL_PATH} \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max_num_seqs 128 \
    --max_num_batched_tokens 32768 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --distributed-executor-backend mp \
    --served-model-name ${SERVED_MODEL_NAME} \
    --compilation-config '{\"splitting_ops\": [\"vllm.unified_attention\",\"vllm.unified_attention_with_output\",\"vllm.unified_attention_with_output_kunlun\",\"vllm.mamba_mixer2\",\"vllm.mamba_mixer\",\"vllm.short_conv\",\"vllm.linear_attention\",\"vllm.plamo2_mamba_mixer\",\"vllm.gdn_attention\",\"vllm.sparse_attn_indexer\"]}' \
    2>&1 | tee ${VLLM_LOG}
"

log "vLLM start command issued (running in background)"
