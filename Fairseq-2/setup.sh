#!/bin/bash
# setup.sh
# Environment setup and pretrained model download for Fairseq Transformer (Colab/Linux)

set -e

# Arguments (with defaults)
BASE_PATH="${1:-/content/drive/MyDrive/Couplet-Data/final/20250711-MiscData/}"
CHECKPOINT_DIR="${2:-checkpoints}"
VENV_PATH="${3:-venv}"

# Create virtual environment and activate
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Upgrade pip and install dependencies
pip install --upgrade pip==23
pip install numpy==1.23.3 fairseq torch cython sacremoses sentencepiece tensorboardX fastBPE

# Download pretrained model and extract
mkdir -p "$CHECKPOINT_DIR"
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz
 tar -xzvf wmt19.en-de.joined-dict.single_model.tar.gz -C "$CHECKPOINT_DIR"

# Move model files to checkpoints
mv "$CHECKPOINT_DIR/wmt19.en-de.joined-dict.single_model/dict.de.txt" "$CHECKPOINT_DIR/" || true
mv "$CHECKPOINT_DIR/wmt19.en-de.joined-dict.single_model/dict.en.txt" "$CHECKPOINT_DIR/" || true
mv "$CHECKPOINT_DIR/wmt19.en-de.joined-dict.single_model/model.pt" "$CHECKPOINT_DIR/" || true
mv "$CHECKPOINT_DIR/wmt19.en-de.joined-dict.single_model/bpecodes" "$CHECKPOINT_DIR/" || true

# Example: Interactive translation
echo "Hello, how are you?" | $VENV_PATH/bin/fairseq-interactive "$CHECKPOINT_DIR" \
    --path "$CHECKPOINT_DIR/model.pt" \
    --source-lang en --target-lang de \
    --tokenizer moses --bpe fastbpe --bpe-codes "$CHECKPOINT_DIR/bpecodes" \
    --beam 5 --remove-bpe

echo "Setup complete."
