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

def generate_translation(model, tokenizer, input_text, max_new_tokens, temperature, top_p):
    """Generate translation for input text with configurable parameters
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        input_text: The text to translate
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        top_p: Top-p sampling parameter (higher = more diverse)
    """
    prompt = format_prompt(input_text)
    
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        top_p=top_p
    )
    
    translation = tokenizer.batch_decode(outputs)[0]
    return translation

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    
    # Model and input configuration
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the fine-tuned model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Classical Chinese text')
    
    # Generation parameters
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Maximum number of tokens to generate (default: 128)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation. Higher values = more random, lower values = more focused. Set to 0 for deterministic output. (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p sampling parameter. Higher values = more diverse output. (default: 0.95)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.temperature < 0:
        parser.error("Temperature must be >= 0")
    if not 0 < args.top_p <= 1:
        parser.error("Top-p must be between 0 and 1")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be > 0")
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Generate translation
    translation = generate_translation(
        model,
        tokenizer,
        args.input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print("\nTranslation:")
    print(translation)

if __name__ == '__main__':
    main()
