import argparse
import torch
from fairseq.models.transformer import TransformerModel
from fairseq.data import Dictionary

def load_model(model_path):
    """Load trained model and dictionary"""
    model = TransformerModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint_best.pt'
    )
    model.eval()
    return model

def translate(model, input_text):
    """Translate a single input sentence"""
    # Tokenize input
    tokens = model.encode(input_text)
    
    # Generate translation
    with torch.no_grad():
        output = model.generate(tokens)
    
    # Decode output
    translation = model.decode(output[0]['tokens'])
    return translation

def main():
    parser = argparse.ArgumentParser(description='Run inference with Fairseq Transformer')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input text to translate')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Generate translation
    translation = translate(model, args.input)
    
    print("\nTranslation:")
    print(translation)

if __name__ == '__main__':
    main()
