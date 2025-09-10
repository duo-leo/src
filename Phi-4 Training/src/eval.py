import argparse
import torch
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
from datasets import Dataset
import numpy as np
from transformers import TextStreamer

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=False,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-4",
    )
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def load_test_data(file_path):
    """Load and parse test data from Excel file"""
    df = pd.read_excel(file_path, engine='openpyxl')
    prompts = []
    
    for prompt_str in df['prompt']:
        try:
            prompt = json.loads(prompt_str)
            prompts.append(prompt)
        except:
            continue
    
    return Dataset.from_list(prompts)

def generate_translation(model, tokenizer, input_text):
    """Generate translation for a single input"""
    messages = [
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
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=2048,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, test_dataset):
    """Evaluate model on test dataset"""
    results = []
    
    for item in tqdm(test_dataset, desc="Evaluating"):
        # Extract user input and expected output
        for i, turn in enumerate(item['conversations']):
            if turn['role'] == 'user':
                input_text = turn['content']
            elif turn['role'] == 'assistant':
                expected_output = turn['content']
                
        # Generate translation
        generated_output = generate_translation(model, tokenizer, input_text)
        
        # Store results
        results.append({
            'input': input_text,
            'expected': expected_output,
            'generated': generated_output
        })
    
    return results

def calculate_metrics(results):
    """Calculate evaluation metrics"""
    # Add your custom metrics here
    # For example: BLEU score, accuracy, etc.
    return {
        'total_samples': len(results),
        # Add more metrics as needed
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Phi-4 model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data Excel file')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load model and test data
    model, tokenizer = load_model(args.model_path)
    test_dataset = load_test_data(args.test_data)
    
    # Run evaluation
    results = evaluate_model(model, tokenizer, test_dataset)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation completed. Results saved to {args.output_file}")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
