#!/bin/bash
#SBATCH --job-name=bd-checkenv
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# Cheap (<5 min) first job: verifies CUDA, bf16 support (expect False on Turing), and that an
# fp16-autocast forward+backward of Qwen2.5-0.5B gives finite loss/grads. Then runs the smoke test.
source scripts/slurm/common.sh
python scripts/check_env.py --model Qwen/Qwen2.5-0.5B
python scripts/smoke_test.py
