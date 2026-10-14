"""The Stan driver's chain-axis handling (needs no Stan install)."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from tests._hb_deps import assert_deps_present

# Only arviz is needed here (the driver imports numpy at module scope and defers the rest), so
# the guard is narrower than the full [bayes] stack the other hb tests check.
try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:  # pragma: no cover - depends on the installed extra
    HAS_ARVIZ = False
assert_deps_present(HAS_ARVIZ)

_PATH = (Path(__file__).parent.parent / "src" / "aind_dynamic_foraging_models"
         / "hierarchical_bayes" / "benchmarks" / "fit_three_level_stan.py")
_SPEC = importlib.util.spec_from_file_location("fit_three_level_stan", _PATH)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _pystan_flat(per_chain):
    """Flatten ``(chain, draw)`` the way PyStan 3 does: draw-major, chain innermost."""
    return np.asarray(per_chain).T.reshape(-1)


class TestChainsFirst(unittest.TestCase):
    """PyStan's flat draw axis has the chain innermost; the driver must undo that."""

    def test_recovers_each_chain(self):
        per_chain = np.array([[0, 1, 2], [100, 101, 102], [200, 201, 202], [300, 301, 302]])
        out = driver.chains_first(_pystan_flat(per_chain), 4, 3)
        np.testing.assert_array_equal(out, per_chain)

    def test_the_chain_major_reshape_is_wrong(self):
        """The mistake this replaces: it mixes the chains, so each 'chain' holds all of them."""
        per_chain = np.array([[0, 1, 2], [100, 101, 102], [200, 201, 202], [300, 301, 302]])
        wrong = _pystan_flat(per_chain).reshape(4, 3)
        self.assertFalse(np.array_equal(wrong, per_chain))
        self.assertEqual(wrong[0].tolist(), [0, 100, 200])   # draw 0 of chains 0, 1, 2 in 'chain 0'

    def test_leading_axes_are_kept(self):
        rng = np.random.default_rng(0)
        per_chain = rng.normal(size=(5, 4, 7))                       # (param, chain, draw)
        flat = np.stack([_pystan_flat(per_chain[p]) for p in range(5)])
        np.testing.assert_array_equal(driver.chains_first(flat, 4, 7), per_chain)

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            driver.chains_first(np.zeros(10), 4, 3)

    @unittest.skipUnless(HAS_ARVIZ, "arviz not installed")
    def test_rhat_detects_separated_chains_only_on_the_right_axis(self):
        """Chains at different levels must give a large R-hat; mixed 'chains' hide it."""
        rng = np.random.default_rng(1)
        per_chain = rng.normal(size=(4, 500)) + np.array([[0.0], [3.0], [6.0], [9.0]])
        flat = _pystan_flat(per_chain)
        right = float(az.rhat(driver.chains_first(flat, 4, 500)))
        wrong = float(az.rhat(flat.reshape(4, 500)))
        self.assertGreater(right, 1.5)
        self.assertLess(wrong, 1.05)


if __name__ == "__main__":
    unittest.main()
