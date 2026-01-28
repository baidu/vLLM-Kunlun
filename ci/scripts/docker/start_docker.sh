#!/usr/bin/env bash
set -euo pipefail

source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Starting docker container: ${DOCKER_NAME}"

if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_NAME}$"; then
  log "Container exists, removing first..."
  docker stop "${DOCKER_NAME}" >/dev/null 2>&1 || true
  docker rm "${DOCKER_NAME}" >/dev/null 2>&1 || true
fi

HOST_CUDA_LIB_PATH=""
for path in "/usr/local/cuda/lib64" /usr/local/cuda-*/lib64; do
  if [ -d "$path" ]; then
    HOST_CUDA_LIB_PATH="$path"
    break
  fi
done

if [ -n "${HOST_CUDA_LIB_PATH}" ]; then
  log "Detected host CUDA lib path: ${HOST_CUDA_LIB_PATH}"
else
  log "Host CUDA lib path not found, will use container CUDA"
fi

# NVIDIA device mapping
DEVICE_ARGS=""
if [ -e "/dev/nvidia0" ]; then
  DEVICE_ARGS="--device /dev/nvidia0:/dev/nvidia0"
  for i in $(seq 1 16); do
    if [ -e "/dev/nvidia${i}" ]; then
      DEVICE_ARGS="${DEVICE_ARGS} --device /dev/nvidia${i}:/dev/nvidia${i}"
    fi
  done
  if [ -e "/dev/nvidia-uvm" ]; then
    DEVICE_ARGS="${DEVICE_ARGS} --device /dev/nvidia-uvm:/dev/nvidia-uvm"
  fi
  if [ -e "/dev/nvidia-modeset" ]; then
    DEVICE_ARGS="${DEVICE_ARGS} --device /dev/nvidia-modeset:/dev/nvidia-modeset"
  fi
else
  log "WARNING: /dev/nvidia0 not found, GPU may not be available"
fi

# Mount nvidia-smi
NVIDIA_BIN=""
if [ -f "/usr/bin/nvidia-smi" ]; then
  NVIDIA_BIN="-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi"
  log "Added nvidia-smi mount"
else
  log "WARNING: nvidia-smi not found on host"
fi

# Mount critical NVIDIA libs
NVIDIA_LIBS=""
if [ -d "/usr/lib64" ]; then
  for lib in libnvidia-ml.so libnvidia-ml.so.1; do
    if [ -f "/usr/lib64/${lib}" ]; then
      NVIDIA_LIBS="${NVIDIA_LIBS} -v /usr/lib64/${lib}:/usr/lib64/${lib}"
    fi
  done
fi

# Ensure libcuda symlink
ln -sf /usr/lib64/libcuda.so.1 /usr/lib64/libcuda.so || true

log "docker run ${IMAGE_NAME}"
docker run \
  -h "$(hostname)" \
  --privileged \
  --net=host \
  --user=root \
  --name="${DOCKER_NAME}" \
  -v /home:/home \
  -v "${WORKSPACE_MOUNT}" \
  -v /ssd2:/ssd2 \
  -v /ssd1:/ssd1 \
  -v /ssd3:/ssd3 \
  -v /dev/shm:/dev/shm \
  -v /usr/lib64/libcuda.so.1:/usr/lib64/libcuda.so.1 \
  -v /usr/lib64/libcuda.so:/usr/lib64/libcuda.so \
  -v /usr/lib64/libnvidia-ml.so.1:/usr/lib64/libnvidia-ml.so.1 \
  -v /usr/lib64/libnvidia-ptxjitcompiler.so.1:/usr/lib64/libnvidia-ptxjitcompiler.so.1 2>/dev/null \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace \
  ${DEVICE_ARGS} \
  ${NVIDIA_BIN} \
  ${NVIDIA_LIBS} \
  --shm-size=16G \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -itd "${IMAGE_NAME}"

log "Container started. Inject conda activate into bashrc"
docker exec "${DOCKER_NAME}" bash -lc "
  echo 'conda activate ${CONDA_ENV}' >> ~/.bashrc
  conda env list || true
"
