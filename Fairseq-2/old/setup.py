# setup.py
"""
Environment setup, dependency installation, and data preparation for Fairseq Transformer.
"""
import subprocess
import sys
import os

# Define paths and configs

# Allow command line arguments for config
import argparse

parser = argparse.ArgumentParser(description="Fairseq Setup")
parser.add_argument('--base_path', type=str, default="/content/drive/MyDrive/Couplet-Data/final/20250711-MiscData/", help='Base path for data')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Checkpoint directory')
parser.add_argument('--venv_path', type=str, default="venv", help='Virtual environment path')
args = parser.parse_args()

BASE_PATH = args.base_path
CHECKPOINT_DIR = args.checkpoint_dir
VENV_PATH = args.venv_path

# Example: Create virtual environment and install dependencies (for UNIX)
def run_shell(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# For Windows, you may need to adapt these commands for PowerShell
# Create virtual environment
run_shell(f"python3.10 -m venv {VENV_PATH}")

# Activate virtual environment and install packages
run_shell(f"{VENV_PATH}\\Scripts\\pip install --upgrade pip==23")
run_shell(f"{VENV_PATH}\\Scripts\\pip install numpy==1.23.3 fairseq torch cython sacremoses sentencepiece tensorboardX fastBPE")

# Data preparation (example: apply BPE, preprocess)
# You can add more logic here as needed

# Download pretrained model and extract
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
run_shell(f"powershell -Command \"Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz -OutFile wmt19.en-de.joined-dict.single_model.tar.gz\"")
run_shell(f"tar -xzvf wmt19.en-de.joined-dict.single_model.tar.gz -C {CHECKPOINT_DIR}")

# Move model files to checkpoints
src_dir = os.path.join(CHECKPOINT_DIR, 'wmt19.en-de.joined-dict.single_model')
for fname in ['dict.de.txt', 'dict.en.txt', 'model.pt', 'bpecodes']:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(CHECKPOINT_DIR, fname)
    if os.path.exists(src):
        os.replace(src, dst)

# Example: Interactive translation
example_cmd = (
    f'echo "Hello, how are you?" | '
    f'{VENV_PATH}\\Scripts\\fairseq-interactive {CHECKPOINT_DIR} '
    f'--path {CHECKPOINT_DIR}\\model.pt '
    '--source-lang en --target-lang de '
    '--tokenizer moses --bpe fastbpe --bpe-codes checkpoints/bpecodes '
    '--beam 5 --remove-bpe'
)
print("Running example interactive translation...")
run_shell(example_cmd)

print("Setup complete.")
