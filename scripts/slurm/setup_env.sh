#!/bin/bash
# One-time environment setup on GAIVI (run on the LOGIN node, not in a job).
#   bash scripts/slurm/setup_env.sh
# Creates ./.venv with Python 3.12 and the pinned stack, with PyTorch built for CUDA 12.1
# (the TITAN RTX node reports CUDA 12.1). Then pre-downloads data + models.
set -euo pipefail
cd "$(dirname "$0")/../.."

# --- Python: adjust to what GAIVI provides (check `module avail`). One of:
# module load python/3.12        # if a module exists
# conda create -y -p ./.conda python=3.12 && conda activate ./.conda
PY=${PY:-python3.12}

$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
python scripts/download_assets.py
python -m pytest -q
echo "Environment ready. Next: sbatch scripts/slurm/check_env.sbatch"
