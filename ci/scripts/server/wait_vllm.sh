#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Waiting for vLLM to be ready: ${VLLM_API_BASE}/v1/models"

docker exec "${DOCKER_NAME}" bash -lc "
  set -e

  for i in {1..90}; do
    if curl -sf ${VLLM_API_BASE}/v1/models >/dev/null; then
      echo 'vLLM is ready'
      tail -n 500 ${VLLM_LOG} || true
      exit 0
    fi
    sleep 5
  done

  echo 'vLLM start failed'
  echo '==== last 500 lines of vllm.log ===='
  tail -n 500 ${VLLM_LOG} || true
  exit 1
"
