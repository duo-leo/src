import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=False,
    )
    
    # Configure tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-4",
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def generate_translation(model, tokenizer, input_text, temperature=1.5, min_p=0.1):
    """Generate translation for input text"""
    messages = [
        {
            "role": "system",
            "content": """You are an AI model trained to act as an expert in Hán Việt (Classical Chinese) and its translation to Modern Vietnamese..."""
        },
        {
            "role": "user",
            "content": input_text
        }
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    outputs = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=2048,
        use_cache=True,
        temperature=temperature,
        min_p=min_p
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned Phi-4 model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input Classical Chinese text to translate')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Temperature for text generation')
    parser.add_argument('--min_p', type=float, default=0.1,
                        help='Minimum probability for text generation')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Generate translation
    translation = generate_translation(
        model,
        tokenizer,
        args.input,
        temperature=args.temperature,
        min_p=args.min_p
    )
    
    print("\nTranslation:")
    print(translation)

if __name__ == "__main__":
    main()
