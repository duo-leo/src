import os
import argparse
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq

def load_prompts_from_excel(file_path):
    """Helper function to load and parse prompts from Excel file"""
    print(f"Loading Excel file: {file_path}")
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"Found {len(df)} rows in Excel file")
    
    prompts = []
    failed_count = 0
    
    for i, prompt_str in enumerate(df['prompt'].tolist()):
        try:
            import json
            prompt = json.loads(prompt_str)
        except json.JSONDecodeError:
            try:
                import ast
                prompt = ast.literal_eval(prompt_str)
            except Exception as e:
                print(f"Failed to parse prompt at row {i+1}: {e}")
                failed_count += 1
                continue
        prompts.append(prompt)
    
    print(f"Successfully loaded {len(prompts)} prompts")
    if failed_count > 0:
        print(f"Failed to parse {failed_count} prompts")
    
    return prompts

def dataset_from_excel(file_path, tokenizer):
    """Load prompts from an Excel file and convert to Hugging Face Dataset"""
    prompts = load_prompts_from_excel(file_path)
    dataset = Dataset.from_list(prompts)
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
    )
    return dataset

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

def train(args):
    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Set up tokenizer with chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-4",
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
    
    # Load datasets
    train_dataset = dataset_from_excel(args.train_data, tokenizer)
    val_dataset = dataset_from_excel(args.valid_data, tokenizer)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=args.num_proc,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
            neftune_noise_alpha=args.neftune_noise_alpha,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0
            )
        ],
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    return trainer_stats

def main():
    parser = argparse.ArgumentParser(description='Train Phi-4 model for translation')
    
    # Model configuration
    parser.add_argument('--base_model', type=str, default="unsloth/Phi-4",
                        help='Base model to use')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Whether to load in 4-bit quantization')
    
    # Data paths
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data Excel file')
    parser.add_argument('--valid_data', type=str, required=True,
                        help='Path to validation data Excel file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saved models')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help='Number of warmup steps')
    parser.add_argument('--num_proc', type=int, default=2,
                        help='Number of preprocessing processes')
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha parameter')
    
    # Evaluation and saving configuration
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Number of steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Number of steps between model saves')
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help='Maximum number of checkpoints to keep')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='Number of evaluations with no improvement after which training will be stopped')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Number of steps between logging')
    parser.add_argument('--neftune_noise_alpha', type=float, default=5,
                        help='NEFTune noise alpha parameter')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()
