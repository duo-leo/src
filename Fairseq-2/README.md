# Fairseq Transformer Workflow

This repository provides a modular workflow for training, inference, and evaluation using Fairseq Transformer models. Each step is separated into its own script for clarity and flexibility.


## Setup

Before running any training or inference, set up your environment and dependencies:

```bash
bash setup.sh <base_path> <checkpoint_dir> <venv_path>
```


## Data Preprocessing

Preprocess your raw data for Fairseq training:

```bash
bash data-preprocess.sh <base_path> <checkpoint_dir> <bpe_codes_path> <venv_path>
```

- `base_path`: Path to your data folder (default: `/content/drive/MyDrive/Couplet-Data/final/20250711-MiscData/`)
- `checkpoint_dir`: Directory for model checkpoints (default: `checkpoints`)
- `bpe_codes_path`: Path to BPE codes file (default: `checkpoints/bpecodes`)
- `venv_path`: Path for Python virtual environment (default: `venv`)

This script will:
- Clone and build FastBPE (if not present)
- Apply BPE to train/valid/test data
- Rename files for Fairseq
- Run `fairseq-preprocess` to binarize data
## Training

Train your model with configurable options:

```bash
bash train.sh <data_bin_path> <save_dir> <restore_file> <base_model> <venv_path>
```

- `data_bin_path`: Path to Fairseq preprocessed data (default: `data-bin/`)
- `save_dir`: Directory to save checkpoints (default: `/content/drive/MyDrive/Couplet-Data/FairSeq/checkpoints/`)
- `restore_file`: Path to checkpoint to restore (default: `/content/drive/MyDrive/Couplet-Data/FairSeq/checkpoints/checkpoint_last.pt`)
- `base_model`: Model architecture (e.g., `transformer_wmt_en_de_big`)
- `venv_path`: Path for Python virtual environment (default: `venv`)

## Inference

Run inference on a single sentence:

```bash
bash infer.sh <checkpoint_path> <data_bin_path> <bpe_codes_path> <sentence> <venv_path>
```

- `checkpoint_path`: Path to trained model checkpoint (default: `checkpoints/checkpoint_best.pt`)
- `data_bin_path`: Path to Fairseq data-bin (default: `data-bin/`)
- `bpe_codes_path`: Path to BPE codes file (default: `checkpoints/bpecodes`)
- `sentence`: Input sentence for translation/inference (default: `不厭湖上月 [SEP] 宛在水中央`)
- `venv_path`: Path for Python virtual environment (default: `venv`)

## Evaluation

Evaluate your model on a dataset:

```bash
bash eval.sh <checkpoint_path> <data_bin_path> <venv_path>
```

- `checkpoint_path`: Path to trained model checkpoint (default: `checkpoints/checkpoint_best.pt`)
- `data_bin_path`: Path to Fairseq data-bin (default: `data-bin/`)
- `venv_path`: Path for Python virtual environment (default: `venv`)

## Notes
- All scripts are designed for Colab/Linux environments. Adjust paths as needed for your setup.
- Ensure you have Python and Fairseq installed, and your virtual environment is activated.
- For batch inference or custom workflows, modify the scripts as needed.

## Example Workflow

1. Setup environment:
   ```bash
   bash setup.sh /content/drive/MyDrive/Couplet-Data/final/20250711-MiscData/ checkpoints venv
   ```
2. Preprocess data:
    ```bash
    bash data-preprocess.sh /content/drive/MyDrive/Couplet-Data/final/20250711-MiscData/ checkpoints checkpoints/bpecodes venv
    ```
2. Train model:
   ```bash
   bash train.sh data-bin/ checkpoints/ checkpoints/checkpoint_last.pt transformer_wmt_en_de_big venv
   ```
3. Inference:
   ```bash
   bash infer.sh checkpoints/checkpoint_best.pt data-bin/ checkpoints/bpecodes "不厭湖上月 [SEP] 宛在水中央" venv
   ```
4. Evaluation:
   ```bash
   bash eval.sh checkpoints/checkpoint_best.pt data-bin/ venv
   ```

---

For more details, see each script's source code and comments.