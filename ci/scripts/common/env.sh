#!/usr/bin/env bash
set -euo pipefail

# static configuration
export DOCKER_NAME="${DOCKER_NAME:-aiak-e2e-singlecard}"
export IMAGE_NAME="${IMAGE_NAME:-iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.32}"

export CONDA_ENV="${CONDA_ENV:-python310_torch25_cuda}"

export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-8356}"
export VLLM_API_BASE="http://127.0.0.1:${VLLM_PORT}"

export MODEL_PATH="${MODEL_PATH:-/ssd3/models/Qwen3-8B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"

export XPU_VISIBLE_DEVICES="${XPU_VISIBLE_DEVICES:-5}"

# Proxy Configuration
export PROXY_URL="${PROXY_URL:-http://agent.baidu.com:8891}"
export NO_PROXY_LIST="${NO_PROXY_LIST:-localhost,127.0.0.1,::1}"

export WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-/home/E2E/workspace:/workspace}"

# Log Path
export VLLM_LOG="${VLLM_LOG:-/workspace/vllm.log}"
export ACC_LOG="${ACC_LOG:-/workspace/evalscope_accuracy_report.log}"
export PERF_LOG="${PERF_LOG:-/workspace/benchmark_performance_report.log}"
