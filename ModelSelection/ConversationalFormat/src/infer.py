import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def load_model(model_path, chat_template="phi-4"):
    """Load the fine-tuned model and tokenizer
    
    Args:
        model_path: Path to the model checkpoint
        chat_template: Chat template to use (e.g., 'phi-4', 'alpaca', 'chatml'). Default: 'phi-4'
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )
    
    return model, tokenizer

def format_conversation(input_text):
    """Format the conversation for inference"""
    return [
        {
            "role": "system",
            "content": "You are an expert in translating Classical Chinese to Modern Vietnamese."
        },
        {
            "role": "user",
            "content": input_text
        }
    ]

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
    conversation = format_conversation(input_text)
    
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        top_p=top_p
    )
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned Phi-4 model')
    
    # Model and input configuration
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the fine-tuned model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Classical Chinese text')
    parser.add_argument('--chat-template', type=str, default='phi-4',
                        choices=['phi-4', 'alpaca', 'chatml', 'zephyr', 'mistral', 'llama-2'],
                        help='Chat template to use for formatting prompts (default: phi-4)')
    
    # Generation parameters
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Maximum number of tokens to generate (default: 256)')
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
    model, tokenizer = load_model(args.model_path, args.chat_template)
    
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
