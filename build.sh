#!/bin/bash

set -euo pipefail

echo "========= build enter ========="

echo "$PATH"
WORK_DIR=$(cd $(dirname $0) && pwd) && cd $WORK_DIR

echo_cmd() {
    echo $1
    $1
}

echo "========= build vllm ========="

echo_cmd "rm -rf output"
echo_cmd "mkdir -p output"

cd ${WORK_DIR}
rm -rf output/.scm/
tar -zcvf ../vllm-kunlun.tar.gz ../vllm-kunlun/
mv ../vllm-kunlun.tar.gz ./output/

echo "========= build exit ========="
