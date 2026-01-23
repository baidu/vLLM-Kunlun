#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Installing evalscope in container ${DOCKER_NAME}"

sudo docker exec "${DOCKER_NAME}" bash -lc "
  set -e
  conda activate ${CONDA_ENV}

  export http_proxy=${PROXY_URL}
  export https_proxy=${PROXY_URL}
  export NO_PROXY=${NO_PROXY_LIST}
  export no_proxy=${NO_PROXY_LIST}

  pip install evalscope
  pip install 'evalscope[perf]'
"
