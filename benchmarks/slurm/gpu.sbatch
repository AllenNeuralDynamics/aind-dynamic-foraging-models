#!/bin/bash
#SBATCH --job-name=hb-bench-gpu
#SBATCH --partition=aind
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/han.hou/.claude/jobs/2bc6425b/tmp/bench/gpu_%j.out
set -e
B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-hb-gpu/bin
cd /home/han.hou/code/aind-dynamic-foraging-models
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$B/python -c "import jax; print('jax devices:', jax.devices())"
$B/python benchmarks/benchmark_stan_vs_numpyro.py \
  --n-sessions 40 --n-trials 650 --num-chains 16 \
  --num-samples 500 --num-warmup 500 --skip-stan \
  --output /home/han.hou/.claude/jobs/2bc6425b/tmp/bench/gpu.json
