#!/bin/bash

# Default script if no argument is provided
TARGET_SCRIPT=${1:-train_gpt_mlx.py}

# Check if the argument is a filename in the experiments directory
if [[ -n "$1" && ! -f "$TARGET_SCRIPT" && -f "experiments/$1" ]]; then
    TARGET_SCRIPT="experiments/$1"
fi

# Verify the file actually exists
if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "Error: $TARGET_SCRIPT not found."
    exit 1
fi

# Extract the filename without path or extension for the RUN_ID
SCRIPT_NAME=$(basename "$TARGET_SCRIPT" .py)

# Configuration for a quick local test on MLX
export RUN_ID="${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S)"
export ITERATIONS=100000000
export TRAIN_BATCH_TOKENS=2097152
export VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=524288

echo "Starting smoke test: $RUN_ID"
echo "Running script: $TARGET_SCRIPT"
echo "-----------------------------------"

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs! Firing up torchrun for distributed training..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS "$TARGET_SCRIPT"
else
    python "$TARGET_SCRIPT"
fi

echo "-----------------------------------"
echo "Test Complete."