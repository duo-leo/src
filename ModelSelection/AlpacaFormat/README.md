# Unsloth Alpaca Format Training

This project fine-tunes the Gemma model for translating Classical Chinese to Modern Vietnamese using the Unsloth library.

## Requirements

- Python 3.8+
- CUDA-compatible GPU
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

The training data should be provided in an Excel file (.xlsx) with the following columns:
- `Chinese`: The source Classical Chinese text
- `Modern`: The target Modern Vietnamese translation

## Usage

### Training

```bash
python src/train.py \
    --base-model "unsloth/gemma-2b" \
    --train-data "path/to/data.xlsx" \
    --output-dir "path/to/output" \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-steps 300
```

### Inference

```bash
python src/infer.py \
    --model-path "path/to/lora_model" \
    --input "Your Classical Chinese text here" \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --top-p 0.95
```

### Evaluation

```bash
python src/eval.py \
    --model-path "path/to/lora_model" \
    --test-data "path/to/test.xlsx" \
    --output-file "predictions.xlsx"
```