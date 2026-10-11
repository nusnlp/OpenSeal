#!/bin/bash

set -e

echo "Starting F1 score processing at $(date)"
echo "Running on host: $(hostname)"

# Define the output directory for logs and results
RESULT_DIR=""

python process_f1_th_zh_scores.py "${RESULT_DIR}" 2>&1 | tee "${RESULT_DIR}/f1_th_zh_scores.log"

echo "Finished at $(date)"