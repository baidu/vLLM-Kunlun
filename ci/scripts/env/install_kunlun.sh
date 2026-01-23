#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Installing vLLM-Kunlun (PR code) in container ${DOCKER_NAME}"

sudo docker exec "${DOCKER_NAME}" bash -lc "
  set -e
  conda activate ${CONDA_ENV}

  cd /workspace

  export http_proxy=${PROXY_URL}
  export https_proxy=${PROXY_URL}

  git config --global --add safe.directory \"${GITHUB_WORKSPACE}\"

  cd \"${GITHUB_WORKSPACE}\"
  echo '===== USING PR CODE ====='
  git rev-parse HEAD
  git log -1 --oneline

  unset http_proxy
  unset https_proxy

  cd vLLM-Kunlun

  pip install -r requirements.txt
  python setup.py build
  python setup.py install

  # patch torch dynamo eval_frame
  cp vllm_kunlun/patches/eval_frame.py /root/miniconda/envs/${CONDA_ENV}/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py

  wget -O xpytorch-cp310-torch251-ubuntu2004-x64.run \"https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/1130/xpytorch-cp310-torch251-ubuntu2004-x64.run?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-12-02T05%3A01%3A27Z%2F-1%2Fhost%2Ff3cf499234f82303891aed2bcb0628918e379a21e841a3fac6bd94afef491ff7\"
  bash xpytorch-cp310-torch251-ubuntu2004-x64.run

  pip install \"https://baidu-kunlun-public.su.bcebos.com/v1/baidu-kunlun-share/1130/xtorch_ops-0.1.2209%2B6752ad20-cp310-cp310-linux_x86_64.whl?authorization=bce-auth-v1%2FALTAKypXxBzU7gg4Mk4K4c6OYR%2F2025-12-05T06%3A18%3A00Z%2F-1%2Fhost%2F14936c2b7e7c557c1400e4c467c79f7a9217374a7aa4a046711ac4d948f460cd\"
  pip install \"https://cce-ai-models.bj.bcebos.com/v1/vllm-kunlun-0.11.0/triton-3.0.0%2Bb2cde523-cp310-cp310-linux_x86_64.whl\"
  pip install \"https://cce-ai-models.bj.bcebos.com/XSpeedGate-whl/release_merge/20251219_152418/xspeedgate_ops-0.0.0-cp310-cp310-linux_x86_64.whl\"

  export NO_PROXY=${NO_PROXY_LIST}
  export no_proxy=${NO_PROXY_LIST}

  chmod +x \"${GITHUB_WORKSPACE}/vLLM-Kunlun/setup_env.sh\"
  source \"${GITHUB_WORKSPACE}/vLLM-Kunlun/setup_env.sh\"
"
