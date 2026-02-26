#!/usr/bin/env bash
set -eo pipefail
bs=""
source ci/scripts/common/env.sh
source ci/scripts/common/log.sh

log "Running performance test via bench"

docker exec "${DOCKER_NAME}" bash -lc "

    source /root/miniconda/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
    #!/bin/bash
    # ==========================================
    # 1. Define test dimensions
    #    (can be easily extended, e.g., add "2048x2048")
    # ==========================================
    DIMENSIONS=("1024x1024")

    # ==========================================
    # 2. Define concurrency generation logic (densification strategy)
    # ============x==============================
    # Use array concatenation to combine different density ranges
    # Syntax: seq [start] [step] [end]
    CONCURRENCIES=(1)

    # ==========================================
    # 3. Automatically assemble test cases
    # ==========================================
    TEST_COMBINATIONS=() # Initialize empty array

    # 🔄 Modified: outer loop over batch size (concurrency), inner loop over dimensions
    for bs in "${CONCURRENCIES[@]}"; do    # ← outer loop: concurrency
        for dim in "${DIMENSIONS[@]}"; do  # ← inner loop: dimensions
            case_str="${bs}x${dim}"
            TEST_COMBINATIONS+=("$case_str")
        done
    done

    # ==========================================
    # 4. (Optional) Print generated cases for sanity check
    # ==========================================
    echo "Generated ${#TEST_COMBINATIONS[@]} test cases in total:"
    echo "${TEST_COMBINATIONS[@]}" # Uncomment if you want to print all cases

    # Progress counters
    TOTAL_TESTS=${#TEST_COMBINATIONS[@]}
    CURRENT_TEST=0

    # Iterate over all test combinations
    for COMBINATION in "${TEST_COMBINATIONS[@]}"; do
        # Parse parameters from combination string
        NUM_PROMPTS=$(echo $COMBINATION | cut -d'x' -f1)
        INPUT_LEN=$(echo $COMBINATION | cut -d'x' -f2)
        OUTPUT_LEN=$(echo $COMBINATION | cut -d'x' -f3)

        # Update progress
        CURRENT_TEST=$((CURRENT_TEST + 1))

        echo "=========================================================="
        echo "Test progress: $CURRENT_TEST / $TOTAL_TESTS"
        echo "Current configuration: concurrency=$NUM_PROMPTS, input_len=$INPUT_LEN, output_len=$OUTPUT_LEN"
        echo "=========================================================="

        #OUTPUT_FILE="$RESULT_DIR/p800_${NUM_PROMPTS}_${INPUT_LEN}_${OUTPUT_LEN}.log"

        # Run benchmark
        python3 -m vllm.entrypoints.cli.main bench serve \
            --host 127.0.0.1 \
            --port ${VLLM_PORT} \
            --backend vllm \
            --model ${SERVED_MODEL_NAME} \
            --dataset-name random \
            --num-prompts $NUM_PROMPTS \
            --random-input-len $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --tokenizer ${MODEL_PATH} \
            --ignore-eos
    done
"
