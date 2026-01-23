#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Stopping docker container: ${DOCKER_NAME}"
sudo docker stop "${DOCKER_NAME}" >/dev/null 2>&1 || true
sudo docker rm "${DOCKER_NAME}" >/dev/null 2>&1 || true
log "Cleanup done"
