# Fairseq Transformer for Translation

This project implements a Fairseq Transformer model for neural machine translation.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required packages (see `src/requirements.txt`)

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

2. Install required packages:
```bash
pip install -r src/requirements.txt
```

## Data Format

The training, validation, and test data should be organized in parallel text files:
- `train.{src,tgt}`: Training data in source and target languages
- `valid.{src,tgt}`: Validation data
- `test.{src,tgt}`: Test data

Each line in these files should contain one sentence, with source and target sentences aligned across files.

## Usage

### Training

```bash
python src/train.py \
    --base-model "facebook/fairseq-transformer" \
    --train-data "path/to/train" \
    --valid-data "path/to/valid" \
    --output-dir "path/to/output" \
    --batch-size 32 \
    --max-epochs 100
```

### Inference

```bash
python src/infer.py \
    --model-path "path/to/checkpoint" \
    --input "Your source text here"
```

### Evaluation

```bash
python src/eval.py \
    --model-path "path/to/checkpoint" \
    --test-data "path/to/test"
```
