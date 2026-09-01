"""Whether the hierarchical-Bayes optional stack is importable, and whether it must be.

The test modules used to guard themselves with a bare ``try/except ImportError`` and skip.
That reads as reasonable -- the extra is optional -- but it means a broken extra reports
``OK (skipped=41)``: every test that touches the sampler silently does not run, and the
only visible trace is a coverage number.

That happened. A ``numpyro<0.20`` cap with an unbounded ``jax`` resolved to a pair numpyro
cannot import, and the suite went green.

So skipping is allowed only where the extra is genuinely absent. Setting
``AIND_HB_REQUIRE_DEPS=1`` -- which the CI job that installs ``[bayes]`` does -- turns an
import failure into a loud error instead.
"""

import os

IMPORT_ERROR = None
try:
    import arviz  # noqa: F401
    import jax  # noqa: F401
    import numpyro  # noqa: F401
except ImportError as error:  # pragma: no cover - depends on the installed extra
    IMPORT_ERROR = error


def assert_deps_present(module_flag):
    """Raise when a module's deps are absent but were required.

    Called at module scope so the failure surfaces as a collection error, not a skip.

    Parameters
    ----------
    module_flag : bool
        The calling module's own "deps imported" flag.
    """
    if module_flag or os.environ.get("AIND_HB_REQUIRE_DEPS") != "1":
        return
    raise ImportError(
        f"AIND_HB_REQUIRE_DEPS=1 but this module's dependencies do not import: "
        f"{IMPORT_ERROR}. Skipping would report OK while testing nothing."
    )
