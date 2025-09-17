#!/bin/bash
# data-preprocess.sh
# Preprocess data for Fairseq training (Linux/WSL)

set -e

# Set base paths (customize as needed)
BASE_PATH="${1:-/content/drive/MyDrive/Couplet-Data/final/20250711-MiscData/}"
CHECKPOINTS_PATH="${2:-checkpoints}"
BPECODES_PATH="${3:-$CHECKPOINTS_PATH/bpecodes}"
VENV_PATH="${4:-venv}"

# Clone and build FastBPE
if [ ! -d "fastBPE" ]; then
    git clone https://github.com/glample/fastBPE.git
    cd fastBPE
    "$VENV_PATH/bin/python" setup.py install
    g++ -std=c++11 -pthread -O3 main.cc -o fastbpe
    mv fastbpe ../"$VENV_PATH/bin/"
    cd ..
fi

# Apply BPE to train/valid/test data
for SPLIT in train valid test; do
    "$VENV_PATH/bin/fastbpe" applybpe "$BASE_PATH${SPLIT}.zh.bpe" "$BASE_PATH${SPLIT}.zh" "$BPECODES_PATH"
    "$VENV_PATH/bin/fastbpe" applybpe "$BASE_PATH${SPLIT}.vi.bpe" "$BASE_PATH${SPLIT}.vi" "$BPECODES_PATH"
    # Rename for Fairseq
    mv "$BASE_PATH${SPLIT}.zh.bpe" "$BASE_PATH${SPLIT}.zh-vi.zh"
    mv "$BASE_PATH${SPLIT}.vi.bpe" "$BASE_PATH${SPLIT}.zh-vi.vi"
    # Print line counts
    wc -l "$BASE_PATH${SPLIT}.zh-vi.zh" "$BASE_PATH${SPLIT}.zh-vi.vi"
done

# Run fairseq-preprocess
"$VENV_PATH/bin/fairseq-preprocess" \
    --source-lang zh --target-lang vi \
    --trainpref "$BASE_PATH/train.zh-vi" \
    --validpref "$BASE_PATH/valid.zh-vi" \
    --testpref "$BASE_PATH/test.zh-vi" \
    --destdir data-bin/ \
    --joined-dictionary \
    --workers 4

echo "Data preprocessing complete."
