# Benchmark job scripts

SLURM scripts for the reference comparison, sized so the numbers mean something.

- `cpu16.sbatch` — 16 cores, so Stan's 16 chains each get their own core. On a 4-core node
  Stan's chains serialise and the comparison understates it fourfold.
- `gpu.sbatch` — one GPU, NumPyro only. Stan has no GPU path for this model, so the GPU leg
  measures what the JAX rewrite buys rather than a like-for-like race.

Both use 40 sessions of 650 trials, the shape of a real subject, with 16 chains and 500
warmup + 500 sampling iterations. That is fewer iterations than the published configuration
(2500 warmup, 5000 samples) so the benchmark finishes in reasonable time; the ratio between
implementations is what these measure, not absolute time to a publishable fit.

Output paths point at a scratch directory and should be edited before reuse.
