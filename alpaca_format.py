from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
DATA_PATH = '/home/clc_hcmus/tanh_nhung/all.xlsx'
OUTPUT_PATH = '/home/clc_hcmus/tanh_nhung/gemma-prediction.xlsx'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
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

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

import pandas as pd
from datasets import Dataset

def create_dataset_from_excel(excel_file_path):
  # Load the Excel file into a pandas DataFrame
  df = pd.read_excel(excel_file_path)

  # Preprocess the data
  df['Chinese'] = df['Chinese'].astype(str).str.replace('\n', ' . ')
  df['Modern'] = df['Modern'].astype(str).str.replace('\n', ' ').str.strip()

  # Create the dataset
  dataset = Dataset.from_pandas(df)

  # Define the instruction and input/output columns
  def format_instruction(example):
      return {
          'instruction': "Translate the Classical Chinese couplet into Modern Vietnamese",
          'input': example['Chinese'],
          'output': example['Modern']
      }

  dataset = dataset.map(format_instruction, remove_columns=['Chinese', 'Sino', 'Modern'])

  return dataset

# Example usage:
# Replace 'your_excel_file.xlsx' with the actual path to your Excel file
dataset = create_dataset_from_excel(DATA_PATH)
dataset = dataset.map(formatting_prompts_func, batched = True,)
print(dataset[5])

train_dataset = dataset.train_test_split(test_size=0.2, seed=42)['train']
eval_dataset = dataset.train_test_split(test_size=0.2, seed=42)['test']
val_dataset = eval_dataset.train_test_split(test_size=0.5, seed=42)['train']
eval_dataset = eval_dataset.train_test_split(test_size=0.5, seed=42)['test']

# Print the sizes of the datasets
print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Evaluation dataset size:", len(eval_dataset))

print(eval_dataset[0])

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import DatasetDict

eval_config = (
    {"eval_steps": 10, "eval_strategy": "steps", "do_eval": True}
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 300,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
        **eval_config,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# prompt: do inference on eval_dataset and output to an xlsx

import pandas as pd
from tqdm import trange

predictions = []
for i in trange(len(eval_dataset)):
    example = eval_dataset[i]
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                example['instruction'],
                example['input'],
                "",  # Leave output blank for generation
            )
        ],
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    prediction = tokenizer.batch_decode(outputs)[0]
    print(prediction)
    print("===")
    predictions.append(prediction)

# Create a DataFrame from the predictions
results_df = pd.DataFrame({'input': eval_dataset['input'],
                         'output': eval_dataset['output'],
                         'prediction': predictions})

# Save the DataFrame to an Excel file
results_df.to_excel(OUTPUT_PATH, index=False)
