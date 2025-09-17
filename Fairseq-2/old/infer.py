# infer.py
"""
Run inference using Fairseq Transformer model.
"""
import subprocess

# Configs
sentence = "不厭湖上月 [SEP] 宛在水中央"

# Allow command line arguments for config
import argparse

parser = argparse.ArgumentParser(description="Fairseq Inference")
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/checkpoint_best.pt", help='Path to model checkpoint')
parser.add_argument('--data_bin_path', type=str, default="data-bin/", help='Path to data-bin directory')
parser.add_argument('--bpe_codes_path', type=str, default="checkpoints/bpecodes", help='Path to BPE codes file')
parser.add_argument('--sentence', type=str, default="不厭湖上月 [SEP] 宛在水中央", help='Input sentence for inference')
args = parser.parse_args()

CHECKPOINT_PATH = args.checkpoint_path
DATA_BIN_PATH = args.data_bin_path
BPE_CODES_PATH = args.bpe_codes_path
sentence = args.sentence

infer_cmd = (
    f'echo "{sentence}" | venv\\Scripts\\fairseq-interactive {DATA_BIN_PATH} '
    f'--path {CHECKPOINT_PATH} '
    '--source-lang zh --target-lang vi '
    '--tokenizer moses --bpe fastbpe '
    f'--bpe-codes {BPE_CODES_PATH} '
    '--beam 5 --remove-bpe'
)

print("Running inference...")
subprocess.run(infer_cmd, shell=True, check=True)
print("Inference complete.")
