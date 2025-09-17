#!/bin/bash
# eval.sh
# Run evaluation using Fairseq Transformer model (Colab/Linux)

set -e

CHECKPOINT_PATH="${1:-checkpoints/checkpoint_best.pt}"
DATA_BIN_PATH="${2:-data-bin/}"
VENV_PATH="${3:-venv}"

source "$VENV_PATH/bin/activate"

$VENV_PATH/bin/fairseq-generate "$DATA_BIN_PATH" \
    --path "$CHECKPOINT_PATH" \
    --batch-size 32 --beam 3 --remove-bpe

echo "Evaluation complete."
