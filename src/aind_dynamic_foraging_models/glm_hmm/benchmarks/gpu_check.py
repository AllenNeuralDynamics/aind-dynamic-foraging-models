"""Check that GPU linear algebra works, and report which CUDA libraries were loaded.

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


def run(label, fn):
    """Print ``label: ok <value>``, or ``label: FAIL <error>``."""
    try:
        print(f"{label}: ok {float(fn()):.6f}")
    except Exception as error:  # report every failure the same way, then carry on
        print(f"{label}: FAIL {type(error).__name__}: {str(error).splitlines()[0][:150]}")


def loaded_libraries():
    """Paths of the CUDA libraries this process has mapped."""
    with open("/proc/self/maps") as maps:
        return sorted({line.split()[-1] for line in maps if any(k in line for k in LIBRARIES)})


def main():
    """Run the checks."""
    print("jax", jax.__version__, jax.devices())
    a = jnp.eye(20) * 2.0 + 0.1
    run("matmul f32", lambda: (a @ a).sum())
    run("solve f32", lambda: jnp.linalg.solve(a, jnp.ones(20)).sum())
    with enable_x64():
        a64 = jnp.eye(20, dtype=jnp.float64) * 2.0 + 0.1
        run("solve f64", lambda: jnp.linalg.solve(a64, jnp.ones(20)).sum())
        run("cholesky f64", lambda: jnp.linalg.cholesky(a64).sum())
        run("batched solve f64", lambda: jnp.linalg.solve(
            jnp.broadcast_to(a64, (10, 20, 20)), jnp.ones((10, 20, 1))).sum())
    print("loaded:", *loaded_libraries(), sep="\n  ")


if __name__ == "__main__":
    main()
