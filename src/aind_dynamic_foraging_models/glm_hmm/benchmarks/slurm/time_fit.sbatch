#!/bin/bash
#SBATCH -J glmhmm_bench
#SBATCH -p aind
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
# Run time_fit.py on a compute node. Submit from the benchmarks directory:
#   CPU: sbatch slurm/time_fit.sbatch PYTHON COHORT.npz parts hurdle 10 5
#   GPU: sbatch --gres=gpu:a100:1 --export=ALL,AIND_GLMHMM_DEVICE=gpu \
#          slurm/time_fit.sbatch PYTHON COHORT.npz parts hurdle 10 5
# PYTHON is an interpreter with the [glmhmm] extra; for a GPU, one whose CUDA solver libraries
# are pinned (RESULTS.md, "Reproduce"). 8 cores and 64 GB match study 14's Beaker template.
set -eu
PYTHON=$1; shift
echo "node $(hostname), $(nproc) cores, device ${AIND_GLMHMM_DEVICE:-cpu}" \
     "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
"$PYTHON" -W ignore "$SLURM_SUBMIT_DIR/time_fit.py" "$@"
