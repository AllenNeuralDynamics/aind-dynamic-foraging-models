"""Whether the GLM-HMM optional stack is importable, and whether it must be.

Same contract as ``_hb_deps``: a module may skip only where the ``glmhmm`` extra is genuinely
absent. ``AIND_GLMHMM_REQUIRE_DEPS=1`` -- set by the CI job that installs the extra -- turns an
import failure into a loud error, so a broken extra cannot report OK while testing nothing.
"""

import os

IMPORT_ERROR = None
try:
    import dynamax  # noqa: F401
    import jax
    import tensorflow_probability.substrates.jax  # noqa: F401

    # Tests also call internals (Newton, E-step) that bypass the package's CPU default; on a
    # GPU host those would hit the cuSOLVER failure the default exists for (model._compute).
    jax.config.update("jax_platforms", "cpu")
except ImportError as error:  # pragma: no cover - depends on the installed extra
    IMPORT_ERROR = error

DEPS_PRESENT = IMPORT_ERROR is None


def assert_deps_present():
    """Raise when the extra is absent but was required.

    Called at module scope so the failure surfaces as a collection error, not a skip.
    """
    if DEPS_PRESENT or os.environ.get("AIND_GLMHMM_REQUIRE_DEPS") != "1":
        return
    raise ImportError(
        f"AIND_GLMHMM_REQUIRE_DEPS=1 but the glmhmm extra does not import: {IMPORT_ERROR}. "
        "Skipping would report OK while testing nothing."
    )
