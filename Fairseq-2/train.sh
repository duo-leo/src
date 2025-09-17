#!/bin/bash
# train.sh
# Train Fairseq Transformer model (Colab/Linux)

set -e

DATA_BIN_PATH="${1:-data-bin/}"
SAVE_DIR="${2:-/content/drive/MyDrive/Couplet-Data/FairSeq/checkpoints/}"
RESTORE_FILE="${3:-/content/drive/MyDrive/Couplet-Data/FairSeq/checkpoints/checkpoint_last.pt}"
BASE_MODEL="${4:-transformer_wmt_en_de_big}"
VENV_PATH="${5:-venv}"

source "$VENV_PATH/bin/activate"

$VENV_PATH/bin/fairseq-train "$DATA_BIN_PATH" \
    --arch "$BASE_MODEL" \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-source-positions 4096 --max-target-positions 4096 \
    --skip-invalid-size-inputs-valid-test \
    --update-freq 16 \
    --save-interval 2 \
    --keep-best-checkpoints 1 \
    --keep-last-epochs 2 \
    --save-dir "$SAVE_DIR" \
    --restore-file "$RESTORE_FILE" \
    --fp16 \
    --max-epoch 30 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 5

echo "Training complete."
