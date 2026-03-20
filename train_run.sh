#!/bin/bash

# Usage:
#   ./train_run.sh                        # defaults to train_gpt_mlx.py
#   ./train_run.sh train_gpt.py           # explicit script
#   ./train_run.sh tlc_vanilla_int8.py    # picks up experiments/ automatically

TARGET_SCRIPT=${1:-train_gpt_mlx.py}

# Allow bare filenames found in experiments/
if [[ -n "$1" && ! -f "$TARGET_SCRIPT" && -f "experiments/$1" ]]; then
    TARGET_SCRIPT="experiments/$1"
fi

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    echo "Error: $TARGET_SCRIPT not found."
    exit 1
fi

SCRIPT_NAME=$(basename "$TARGET_SCRIPT" .py)

# -------------------------------------------------------
# DATA / TOKENIZER (override via env if needed)
# -------------------------------------------------------
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# -------------------------------------------------------
# RUN CONFIGURATION
# -------------------------------------------------------
export RUN_ID="${RUN_ID:-${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S)}"
export ITERATIONS="${ITERATIONS:-1000000000}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
# Don't override TRAIN_BATCH_TOKENS here — let each script use its own default.

echo "Starting run: $RUN_ID"
echo "Script:       $TARGET_SCRIPT"
echo "Data:         $DATA_PATH"
echo "Tokenizer:    $TOKENIZER_PATH"
echo "-----------------------------------"

# -------------------------------------------------------
# GPU DETECTION & LAUNCH
# -------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi

# Always use torchrun (even for 1 GPU) so RANK/WORLD_SIZE/LOCAL_RANK are set
# correctly and the distributed code paths behave deterministically.
echo "Detected $NUM_GPUS GPU(s) — launching via torchrun --nproc_per_node=$NUM_GPUS"
torchrun --standalone --nproc_per_node=$NUM_GPUS "$TARGET_SCRIPT"

echo "-----------------------------------"
echo "Run complete."