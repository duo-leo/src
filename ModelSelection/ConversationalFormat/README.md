# Model Training with Conversational Format

This project fine-tunes the models with Conversational Format for Classical Chinese translation using a conversational format.

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

The training, validation, and test data should be provided in Excel files (.xlsx) with a 'prompt' column containing JSON strings in the following format:

```json
{
    "conversations": [
        {
            "role": "system",
            "content": "System prompt describing the translation task..."
        },
        {
            "role": "user",
            "content": "Classical Chinese text to translate"
        },
        {
            "role": "assistant",
            "content": "Modern Vietnamese translation"
        }
    ]
}
```

## Directory Structure

```
data/
├── train.xlsx
├── valid.xlsx
└── test.xlsx
```

## Usage

### Training

```bash
python src/train.py \
    --base-model "unsloth/Phi-4" \
    --train-data "path/to/train.xlsx" \
    --valid-data "path/to/valid.xlsx" \
    --output-dir "path/to/output" \
    --batch-size 4 \
    --epochs 4
```

### Inference

```bash
# Basic usage
python src/infer.py \
    --model-path "path/to/lora_model" \
    --input "Your Classical Chinese text here" \
    --chat-template "phi-4" \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --top-p 0.95
```

### Evaluation

```bash
python src/eval.py \
    --model-path "path/to/lora_model" \
    --test-data "path/to/test.xlsx" \
    --output-file "results.xlsx"
```
