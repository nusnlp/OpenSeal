#!/bin/bash

CHECKPOINT_DIRS=(
    "aisingapore/Llama-SEA-LION-v3.5-8B-R"
)

RUN_NAME="translation-only"

BASE_MODEL=true

TASK_AND_NUM_EXAMPLES=(
    "translation_only 5"
)

for TASK_AND_NUM in "${TASK_AND_NUM_EXAMPLES[@]}"; do
    TASK=$(echo $TASK_AND_NUM | cut -d' ' -f1)
    NUM_EXAMPLES=$(echo $TASK_AND_NUM | cut -d' ' -f2)

    echo "Running evaluation for task: $TASK with $NUM_EXAMPLES examples"

    OUTPUT_DIR="/outputs/${RUN_NAME}"

    POSTFIX=""
    if [ "$BASE_MODEL" = true ]; then
        POSTFIX="basemodel"
    else
        POSTFIX="instructed"
    fi 

    if [ "$NUM_EXAMPLES" -gt 0 ]; then
        POSTFIX="${POSTFIX}_${NUM_EXAMPLES}shot"
    else
        POSTFIX="${POSTFIX}_0shot"
    fi

    OUTPUT_DIR="${OUTPUT_DIR}/${POSTFIX}"

    echo "Output directory: $OUTPUT_DIR"

    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    for CHECKPOINT_DIR in "${CHECKPOINT_DIRS[@]}"; do
        echo "Evaluating checkpoint directory: $CHECKPOINT_DIR"

        qsub -v CHECKPOINT_DIR="$CHECKPOINT_DIR",CHECKPOINT_NAME="$(basename $CHECKPOINT_DIR)",OUTPUT_DIR="$OUTPUT_DIR/$TASK",TASK="$TASK",NUM_EXAMPLES="$NUM_EXAMPLES",BASE_MODEL="$BASE_MODEL" \
        -N seahelm_eval_$(basename $CHECKPOINT_DIR)_$TASK \
        mass_evaluation.pbs

        sleep 1
    done

    sleep 1
done
