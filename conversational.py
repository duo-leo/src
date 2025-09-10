from unsloth import FastLanguageModel  # FastVisionModel for LLMs
import torch
import os

# Clear GPU cache first
torch.cuda.empty_cache()

# Configuration
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = False  # Use 4bit quantization to reduce memory usage

# Check if local model exists, otherwise use Hugging Face model name
model_path = "/home/clc_hcmus/.cache/huggingface/hub/models--unsloth--Phi-4/snapshots/c6220bde10fff762dbd72c3331894aa4cade249d"
if not os.path.exists(model_path):
    model_path = "unsloth/Phi-4"

print(f"Loading model from: {model_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    device_map="auto",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
if hasattr(model, 'to'):
    model = model.to('cuda')

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return { "text" : texts, }
pass

DATA_DIR = "DATA/"

import pandas as pd
import json
from datasets import Dataset
from tqdm import tqdm
from unsloth.chat_templates import standardize_sharegpt

def load_prompts_from_excel(file_path):
    """Helper function to load and parse prompts from Excel file with progress indicator"""
    print(f"Loading Excel file: {file_path}")
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"Found {len(df)} rows in Excel file")

    prompts = []
    failed_count = 0

    # Use tqdm for progress bar
    for i, prompt_str in enumerate(tqdm(df['prompt'].tolist(), desc="Parsing prompts")):
        try:
            # Try standard JSON parsing first
            prompt = json.loads(prompt_str)
        except json.JSONDecodeError:
            try:
                # Fallback to ast.literal_eval for single-quoted strings
                import ast
                prompt = ast.literal_eval(prompt_str)
            except Exception as e:
                print(f"Failed to parse prompt at row {i+1}: {e}")
                # print(f"Original string: {prompt_str}")
                print("=" * 40)
                failed_count += 1
                continue
        prompts.append(prompt)

    print(f"Successfully loaded {len(prompts)} prompts")
    if failed_count > 0:
        print(f"Failed to parse {failed_count} prompts")

    return prompts

def dataset_from_excel(file_path):
    """Load prompts from an Excel file and convert to Hugging Face Dataset"""
    prompts = load_prompts_from_excel(file_path)
    # prompts = prompts[:100]  # Limit to 1000 prompts for training
    dataset = Dataset.from_list(prompts)
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    return dataset

test_dataset = dataset_from_excel(f"{DATA_DIR}test.xlsx")
val_dataset = dataset_from_excel(f"{DATA_DIR}valid.xlsx")
train_dataset = dataset_from_excel(f"{DATA_DIR}train.xlsx")

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_dataset[5]["conversations"]

train_dataset[5]["text"]

from trl import SFTConfig, SFTTrainer

from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,

    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(

        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 4, # Set this for 1 full training run.
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        neftune_noise_alpha=5,

        eval_strategy="steps",
        eval_steps=500,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
    callbacks=[
      EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)
  ],
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

tokenizer.decode(trainer.train_dataset[5]["input_ids"])

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

trainer_stats = trainer.train()
print(trainer_stats)
model.save_pretrained("lora_model-phi-250625")
tokenizer.save_pretrained("lora_model-phi-250625")