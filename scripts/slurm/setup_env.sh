#!/bin/bash
# One-time environment setup on GAIVI (run on the LOGIN node, not in a job).
#   bash scripts/slurm/setup_env.sh
# Creates ./.venv with Python 3.12 and the pinned stack, with PyTorch built for CUDA 12.1
# (the TITAN RTX node reports CUDA 12.1). Then pre-downloads data + models.
set -euo pipefail
cd "$(dirname "$0")/../.."

# --- Python 3.12. GAIVI's system python3 is 3.9.1 and `module avail` lists no python/cuda
# modules (checked Nov 12), so we use the cluster's conda (/apps/anaconda3) to get a private
# Python 3.12 in ./.conda, and build the venv from it. Override with PY=/path/to/python3.12.
if [ -z "${PY:-}" ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
  else
    [ -x ./.conda/bin/python ] || conda create -y -p ./.conda python=3.12
    PY=./.conda/bin/python
  fi
fi
$PY --version

$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
python scripts/download_assets.py
python -m pytest -q
echo "Environment ready. Next: sbatch scripts/slurm/check_env.sbatch"
