#!/bin/bash
# infer.sh
# Run inference using Fairseq Transformer model (Colab/Linux)

set -e

CHECKPOINT_PATH="${1:-checkpoints/checkpoint_best.pt}"
DATA_BIN_PATH="${2:-data-bin/}"
BPE_CODES_PATH="${3:-checkpoints/bpecodes}"
SENTENCE="${4:-不厭湖上月 [SEP] 宛在水中央}"
VENV_PATH="${5:-venv}"

source "$VENV_PATH/bin/activate"

echo "$SENTENCE" | $VENV_PATH/bin/fairseq-interactive "$DATA_BIN_PATH" \
    --path "$CHECKPOINT_PATH" \
    --source-lang zh --target-lang vi \
    --tokenizer moses --bpe fastbpe --bpe-codes "$BPE_CODES_PATH" \
    --beam 5 --remove-bpe

echo "Inference complete."
