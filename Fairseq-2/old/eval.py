# eval.py
"""
Run evaluation using Fairseq Transformer model.
"""
import subprocess

# Configs

# Allow command line arguments for config
import argparse

parser = argparse.ArgumentParser(description="Fairseq Evaluation")
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/checkpoint_best.pt", help='Path to model checkpoint')
parser.add_argument('--data_bin_path', type=str, default="data-bin/", help='Path to data-bin directory')
args = parser.parse_args()

CHECKPOINT_PATH = args.checkpoint_path
DATA_BIN_PATH = args.data_bin_path

# Evaluation command
eval_cmd = (
    f'venv\\Scripts\\fairseq-generate {DATA_BIN_PATH} '
    f'--path {CHECKPOINT_PATH} '
    '--batch-size 32 --beam 3 --remove-bpe'
)

print("Running evaluation...")
subprocess.run(eval_cmd, shell=True, check=True)
print("Evaluation complete.")
