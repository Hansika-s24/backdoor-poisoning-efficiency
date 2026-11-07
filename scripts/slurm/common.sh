# Sourced by every sbatch script: activate env, set caches, log provenance.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # use the pre-downloaded cache; set 0 if nodes have internet
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
echo "job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
