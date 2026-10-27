#!/bin/bash
#SBATCH -J glmhmm_gpu_check
#SBATCH -p aind
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH --ntasks=1
# Run gpu_check.py on a GPU node. Submit from the benchmarks directory, choosing the GPU type:
#   sbatch --gres=gpu:v100:1 slurm/gpu_check.sbatch PYTHON
set -eu
echo "node $(hostname): $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
"$1" -W ignore "$SLURM_SUBMIT_DIR/gpu_check.py"
