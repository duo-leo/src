import argparse
import torch
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm

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

def load_test_data(file_path):
    """Load test data from Excel file"""
    df = pd.read_excel(file_path)
    df['Chinese'] = df['Chinese'].astype(str).str.replace('\n', ' . ')
    df['Modern'] = df['Modern'].astype(str).str.replace('\n', ' ').str.strip()
    return df

def evaluate(model, tokenizer, test_data):
    """Evaluate model on test dataset"""
    predictions = []
    
    for _, row in tqdm(test_data.iterrows(), desc="Generating translations", total=len(test_data)):
        input_text = row['Chinese']
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
        
        prediction = tokenizer.batch_decode(outputs)[0]
        predictions.append(prediction)
    
    results_df = pd.DataFrame({
        'input': test_data['Chinese'],
        'reference': test_data['Modern'],
        'prediction': predictions
    })
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Gemma model')
    
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
