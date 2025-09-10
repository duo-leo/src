# Phi-4 Training for Classical Chinese Translation

This project fine-tunes the Phi-4 model for translating Classical Chinese to Modern Vietnamese.

## Requirements

- Python 3.10+
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

The training, validation, and test datasets should be Excel files (.xlsx) with the following format:

### Excel File Structure

Each Excel file should contain a single sheet with a 'prompt' column. Each cell in the 'prompt' column should contain a JSON string with the following structure:

```json
{
    "conversations": [
        {
            "role": "system",
            "content": "System prompt describing the task..."
        },
        {
            "role": "user",
            "content": "[translation/couplet] Your Classical Chinese text here..."
        },
        {
            "role": "assistant",
            "content": "Translation and explanation in Modern Vietnamese..."
        }
    ]
}
```

### Required Fields

- conversations: Array of conversation turns
  - role: One of "system", "user", or "assistant"
  - content: The actual content of the message

### Example

prompt column:
```json
{
    "conversations": [
        {
            "role": "system",
            "content": "You are an AI model trained to act as an expert in Hán Việt..."
        },
        {
            "role": "user",
            "content": "[translation/couplet] Help me translate: 自古留傳靈跡寺..."
        },
        {
            "role": "assistant",
            "content": "<result>Từ xưa chùa linh tích còn truyền...</result>"
        }
    ]
}
```


## Usage

### Training

```bash
python src/train.py \
    --base_model "unsloth/Phi-4" \
    --train_data "path/to/train.xlsx" \
    --valid_data "path/to/valid.xlsx" \
    --output_dir "path/to/output" \
    --epochs 4 \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```

### Inference

```bash
python src/infer.py \
    --model_path "path/to/checkpoint" \
    --input "Your Classical Chinese text here" \
    --temperature 1.5 \
    --min_p 0.1
```

### Evaluation

```bash
python src/eval.py \
    --model_path "path/to/checkpoint" \
    --test_data "path/to/test.xlsx"
```