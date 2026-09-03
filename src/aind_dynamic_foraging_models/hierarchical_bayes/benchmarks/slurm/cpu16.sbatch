#!/bin/bash
#SBATCH --job-name=hb-bench-cpu16
#SBATCH --partition=aind
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/han.hou/.claude/jobs/2bc6425b/tmp/bench/cpu16_%j.out
set -e
B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-stan/bin
export PATH="$B:$PATH" CXX="$B/g++" CC="$B/gcc"
export XLA_FLAGS="--xla_force_host_platform_device_count=16"
cd /home/han.hou/code/aind-dynamic-foraging-models
echo "=== node $(hostname)  cores $SLURM_CPUS_ON_NODE ==="
$B/python src/aind_dynamic_foraging_models/hierarchical_bayes/benchmarks/benchmark_stan_vs_numpyro.py \
  --n-sessions 40 --n-trials 650 --num-chains 16 \
  --num-samples 500 --num-warmup 500 \
  --output /home/han.hou/.claude/jobs/2bc6425b/tmp/bench/cpu16.json
