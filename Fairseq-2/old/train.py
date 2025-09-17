# train.py
"""
Train Fairseq Transformer model using specified configs.
"""
import subprocess

# Configs

# Allow command line arguments for config
import argparse

parser = argparse.ArgumentParser(description="Fairseq Training")
parser.add_argument('--data_bin_path', type=str, default="data-bin/", help='Path to data-bin directory')
parser.add_argument('--save_dir', type=str, default="/content/drive/MyDrive/Couplet-Data/FairSeq/checkpoints/", help='Directory to save checkpoints')
parser.add_argument('--restore_file', type=str, default="/content/drive/MyDrive/Couplet-Data/FairSeq/checkpoints/checkpoint_last.pt", help='Checkpoint file to restore')
parser.add_argument('--base_model', type=str, default="transformer_wmt_en_de_big", help='Base model architecture')
args = parser.parse_args()

DATA_BIN_PATH = args.data_bin_path
SAVE_DIR = args.save_dir
RESTORE_FILE = args.restore_file
BASE_MODEL = args.base_model

train_cmd = (
    f"venv\Scripts\fairseq-train {DATA_BIN_PATH} "
    f"--arch {BASE_MODEL} "
    "--share-all-embeddings "
    "--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 "
    "--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 "
    "--dropout 0.3 --weight-decay 0.0001 "
    "--criterion label_smoothed_cross_entropy --label-smoothing 0.1 "
    "--max-tokens 4096 "
    "--max-source-positions 4096 --max-target-positions 4096 "
    "--skip-invalid-size-inputs-valid-test "
    "--update-freq 16 "
    "--save-interval 2 "
    "--keep-best-checkpoints 1 "
    "--keep-last-epochs 2 "
    f"--save-dir {SAVE_DIR} "
    f"--restore-file {RESTORE_FILE} "
    "--fp16 "
    "--max-epoch 30 "
    "--eval-bleu "
    "--eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' "
    "--eval-bleu-detok moses "
    "--eval-bleu-remove-bpe "
    "--eval-bleu-print-samples "
    "--best-checkpoint-metric bleu --maximize-best-checkpoint-metric "
    "--patience 5"
)

print("Running training...")
subprocess.run(train_cmd, shell=True, check=True)
print("Training complete.")
