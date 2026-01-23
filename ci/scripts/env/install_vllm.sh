#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Installing vLLM==0.11.0 in container ${DOCKER_NAME}"

sudo docker exec "${DOCKER_NAME}" bash -lc "
  set -e
  conda activate ${CONDA_ENV}
  pip uninstall -y vllm || true
  env | grep -i proxy || true
  pip install vllm==0.11.0 --no-build-isolation --no-deps --index-url https://pip.baidu-int.com/simple/
  python -c 'import vllm; print(\"vllm version:\", vllm.__version__)'
"
