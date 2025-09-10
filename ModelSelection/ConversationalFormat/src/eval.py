import argparse
import torch
import pandas as pd
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-4",
    )
    
    return model, tokenizer

def load_test_data(file_path):
    """Load and parse test data from Excel file"""
    print(f"Loading test data from: {file_path}")
    df = pd.read_excel(file_path)
    
    prompts = []
    failed_count = 0
    
    for i, prompt_str in enumerate(tqdm(df['prompt'], desc="Parsing prompts")):
        try:
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

def evaluate(model, tokenizer, test_data):
    """Evaluate model on test dataset"""
    results = []
    
    for prompt in tqdm(test_data, desc="Generating translations"):
        input_text = None
        reference = None
        
        # Extract input and reference from conversations
        for turn in prompt['conversations']:
            if turn['role'] == 'user':
                input_text = turn['content']
            elif turn['role'] == 'assistant':
                reference = turn['content']
        
        if input_text and reference:
            # Format conversation and generate translation
            conversation = format_conversation(input_text)
            
            inputs = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                use_cache=True,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'input': input_text,
                'reference': reference,
                'prediction': prediction
            })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Phi-4 model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the fine-tuned model')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data Excel file')
    parser.add_argument('--output-file', type=str, default='evaluation_results.xlsx',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load model and test data
    model, tokenizer = load_model(args.model_path)
    test_data = load_test_data(args.test_data)
    
    # Run evaluation
    results_df = evaluate(model, tokenizer, test_data)
    
    # Save results
    results_df.to_excel(args.output_file, index=False)
    print(f"\nEvaluation completed. Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
