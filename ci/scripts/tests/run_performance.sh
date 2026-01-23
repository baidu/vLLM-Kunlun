#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Running performance test via evalscope"

sudo docker exec "${DOCKER_NAME}" bash -lc "
  set -e
  rm -f ${PERF_LOG}

  export http_proxy=${PROXY_URL}
  export https_proxy=${PROXY_URL}
  export NO_PROXY=${NO_PROXY_LIST}
  export no_proxy=${NO_PROXY_LIST}

  evalscope perf \
    --model ${SERVED_MODEL_NAME} \
    --url \"http://localhost:${VLLM_PORT}/v1/chat/completions\" \
    --parallel 5 \
    --number 20 \
    --api openai \
    --dataset openqa \
    --stream 2>&1 | tee ${PERF_LOG}
"
