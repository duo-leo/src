import argparse
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import pandas as pd

def create_dataset_from_excel(excel_file_path):
    """Create dataset from Excel file"""
    df = pd.read_excel(excel_file_path)
    
    # Preprocess the data
    df['Chinese'] = df['Chinese'].astype(str).str.replace('\n', ' . ')
    df['Modern'] = df['Modern'].astype(str).str.replace('\n', ' ').str.strip()
    
    # Create the dataset
    dataset = Dataset.from_pandas(df)
    
    # Format instruction
    def format_instruction(example):
        return {
            'instruction': "Translate the Classical Chinese couplet into Modern Vietnamese",
            'input': example['Chinese'],
            'output': example['Modern']
        }
    
    dataset = dataset.map(format_instruction, remove_columns=['Chinese', 'Sino', 'Modern'])
    return dataset

def format_alpaca_prompt(instruction, input_text, output=""):
    """Format the prompt in Alpaca style"""
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    return prompt

def formatting_prompts_func(examples, tokenizer):
    """Format prompts for training"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = format_alpaca_prompt(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    
    return {"text": texts}

def train(args):
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    
    # Load dataset
    dataset = create_dataset_from_excel(args.train_data)
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )
    
    # Split dataset
    splits = dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_dataset = splits['train']
    temp_eval = splits['test']
    eval_splits = temp_eval.train_test_split(test_size=0.5, seed=args.seed)
    val_dataset = eval_splits['train']
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.num_proc,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
            eval_steps=args.eval_steps,
            evaluation_strategy="steps",
        ),
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model
    model.save_pretrained(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    return trainer_stats

def main():
    parser = argparse.ArgumentParser(description='Train Gemma model for translation')
    
    # Model configuration
    parser.add_argument('--base-model', type=str, default="unsloth/gemma-2b",
                        help='Base model to use')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Use 4-bit quantization')
    
    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA attention dimension')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha parameter')
    
    # Training configuration
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data Excel file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for saved models')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum number of training steps')
    parser.add_argument('--warmup-steps', type=int, default=5,
                        help='Number of warmup steps')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--logging-steps', type=int, default=1,
                        help='Logging steps')
    parser.add_argument('--eval-steps', type=int, default=10,
                        help='Evaluation steps')
    parser.add_argument('--num-proc', type=int, default=2,
                        help='Number of preprocessing workers')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Train the model
    train(args)

if __name__ == '__main__':
    main()
