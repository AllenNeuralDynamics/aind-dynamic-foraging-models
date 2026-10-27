"""Check that linear algebra runs on the GPU, and report the CUDA libraries loaded.

    python gpu_check.py

The GLM-HMM's Newton M-step needs ``jnp.linalg.solve`` in float64. With the CUDA libraries pip
resolves for jax-cuda12-plugin 0.6.1, cuSOLVER 11.7.5 (CUDA 12.9) over cuBLAS 12.8, every solve
fails with ``gpusolverDnCreate ... cuSolver internal error``. The library paths printed at the
end show what was actually loaded. Read them before trusting a result: a venv layered over
another environment can load that environment's copy instead of its own.
"""

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64

LIBRARIES = ("libcusolver", "libcublas", "libnvJitLink", "libcusparse", "libcudart")


def run(label, fn, device):
    """Print ``label: ok <value>`` if ``fn`` ran on ``device``, else ``label: FAIL <why>``."""
    try:
        result = fn()
        if result.devices() != {device}:
            print(f"{label}: FAIL ran on {sorted(map(str, result.devices()))}, not {device}")
            return
        print(f"{label}: ok {float(result):.6f} on {device}")
    except Exception as error:  # report every failure the same way, then carry on
        print(f"{label}: FAIL {type(error).__name__}: {str(error).splitlines()[0][:150]}")


def loaded_libraries():
    """Paths of the CUDA libraries this process has mapped."""
    with open("/proc/self/maps") as maps:
        return sorted({line.split()[-1] for line in maps if any(k in line for k in LIBRARIES)})


def main():
    """Run every check on the first GPU; exit with an error if jax sees none."""
    print("jax", jax.__version__, jax.devices())
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError as error:
        raise SystemExit(f"FAIL: jax sees no GPU ({error}), so there is nothing to check")
    with jax.default_device(gpu):
        a = jnp.eye(20) * 2.0 + 0.1
        run("matmul f32", lambda: (a @ a).sum(), gpu)
        run("solve f32", lambda: jnp.linalg.solve(a, jnp.ones(20)).sum(), gpu)
        with enable_x64():
            a64 = jnp.eye(20, dtype=jnp.float64) * 2.0 + 0.1
            run("solve f64", lambda: jnp.linalg.solve(a64, jnp.ones(20)).sum(), gpu)
            run("cholesky f64", lambda: jnp.linalg.cholesky(a64).sum(), gpu)
            run("batched solve f64", lambda: jnp.linalg.solve(
                jnp.broadcast_to(a64, (10, 20, 20)), jnp.ones((10, 20, 1))).sum(), gpu)
    print("loaded:", *loaded_libraries(), sep="\n  ")


if __name__ == "__main__":
    main()
