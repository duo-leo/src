import argparse
import torch
import sacrebleu
from fairseq.models.transformer import TransformerModel
from tqdm import tqdm

def load_model(model_path):
    """Load trained model"""
    model = TransformerModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint_best.pt'
    )
    model.eval()
    return model

def load_test_data(test_data_path):
    """Load source and reference sentences"""
    with open(f"{test_data_path}.src", 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(f"{test_data_path}.tgt", 'r', encoding='utf-8') as f:
        ref_lines = f.readlines()
    
    return src_lines, ref_lines

def evaluate(model, src_lines, ref_lines):
    """Evaluate model performance"""
    translations = []
    
    for src in tqdm(src_lines, desc="Translating"):
        # Generate translation
        with torch.no_grad():
            tokens = model.encode(src.strip())
            output = model.generate(tokens)
            translation = model.decode(output[0]['tokens'])
            translations.append(translation)
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(translations, [ref_lines])
    
    return {
        'bleu': bleu.score,
        'translations': translations
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate Fairseq Transformer model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data (without extension)')
    parser.add_argument('--output-file', type=str, default='evaluation_results.txt',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load model and test data
    model = load_model(args.model_path)
    src_lines, ref_lines = load_test_data(args.test_data)
    
    # Run evaluation
    results = evaluate(model, src_lines, ref_lines)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score: {results['bleu']:.2f}\n\n")
        f.write("Translations:\n")
        for src, ref, hyp in zip(src_lines, ref_lines, results['translations']):
            f.write(f"\nSource: {src.strip()}")
            f.write(f"Reference: {ref.strip()}")
            f.write(f"Hypothesis: {hyp.strip()}\n")
    
    print(f"\nEvaluation completed. Results saved to {args.output_file}")
    print(f"BLEU Score: {results['bleu']:.2f}")

if __name__ == '__main__':
    main()
