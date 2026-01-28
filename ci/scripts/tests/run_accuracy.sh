#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Running accuracy test via evalscope"

docker exec "${DOCKER_NAME}" bash -lc "
  set -e
  rm -f ${ACC_LOG}

  export http_proxy=${PROXY_URL}
  export https_proxy=${PROXY_URL}
  export NO_PROXY=${NO_PROXY_LIST}
  export no_proxy=${NO_PROXY_LIST}

  evalscope eval \
    --model ${SERVED_MODEL_NAME} \
    --api-url http://localhost:${VLLM_PORT}/v1 \
    --datasets gsm8k arc \
    --limit 10 2>&1 | tee ${ACC_LOG}
"
