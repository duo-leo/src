import argparse
import torch
from unsloth import FastLanguageModel

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    return model, tokenizer

def format_prompt(input_text):
    """Format the input prompt"""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Translate the Classical Chinese couplet into Modern Vietnamese

### Input:
{input_text}

### Response:
"""

def generate_translation(model, tokenizer, input_text):
    """Generate translation for input text"""
    prompt = format_prompt(input_text)
    
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True
    )
    
    translation = tokenizer.batch_decode(outputs)[0]
    return translation

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned Gemma model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the fine-tuned model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Classical Chinese text')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Generate translation
    translation = generate_translation(model, tokenizer, args.input)
    
    print("\nTranslation:")
    print(translation)

if __name__ == '__main__':
    main()
