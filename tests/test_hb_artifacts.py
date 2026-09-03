"""Persisting a fit so it can be reused without refitting."""

import json
import tempfile
import unittest

from tests._hb_deps import assert_deps_present
from pathlib import Path

import numpy as np

try:
    import jax
    from numpyro.infer import MCMC, NUTS

    from aind_dynamic_foraging_models.hierarchical_bayes.artifacts import (
        POPULATION_SITES,
        load_population,
        save_fit,
        summarise_diagnostics,
    )
    from aind_dynamic_foraging_models.hierarchical_bayes.model import (
        hattori2019_three_level,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

# A broken extra must not report OK by skipping every test that touches it. With
# AIND_HB_REQUIRE_DEPS=1 -- which the CI job that installs [bayes] sets -- a failed
# import becomes an error here instead of a run of silent skips.
assert_deps_present(HAS_JAX)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestSaveFit(unittest.TestCase):
    """Round-tripping a fit through netCDF."""

    @classmethod
    def setUpClass(cls):
        """Run a very small three-level fit on random data."""
        rng = np.random.default_rng(0)
        cls.choices = rng.integers(0, 2, (3, 4, 60))
        cls.rewards = rng.integers(0, 2, (3, 4, 60)).astype(float)
        mcmc = MCMC(
            NUTS(hattori2019_three_level),
            num_warmup=40, num_samples=40, num_chains=1, progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0), cls.choices, cls.rewards)
        cls.mcmc = mcmc

    def test_saves_population_and_subject_sites(self):
        """Population and subject draws are written, not just their means."""
        with tempfile.TemporaryDirectory() as tmp:
            result = save_fit(self.mcmc, tmp, name="t", meta={"seed": 0})
            self.assertTrue(Path(result["netcdf"]).exists())

            record = json.loads(Path(result["json"]).read_text())
            for site in POPULATION_SITES:
                self.assertIn(site, record["sites_saved"])
            self.assertIn("mu_p", record["sites_saved"])
            self.assertEqual(record["n_draws"], 40)

    def test_session_sites_are_opt_in(self):
        """Bulky session-level sites are excluded unless asked for."""
        with tempfile.TemporaryDirectory() as tmp:
            without = save_fit(self.mcmc, tmp, name="a")
            with_sessions = save_fit(
                self.mcmc, tmp, name="b", include_session_sites=True
            )
            a = json.loads(Path(without["json"]).read_text())["sites_saved"]
            b = json.loads(Path(with_sessions["json"]).read_text())["sites_saved"]
            self.assertNotIn("session_log_lik", a)
            self.assertIn("session_log_lik", b)

    def test_session_sites_carry_theta_raw_for_replay(self):
        """A one_stage fit persists what a per-session replay reconstructs from.

        `hattori2019_three_level` registers exactly one session-level site,
        `session_log_lik`; the five named parameters in SESSION_SITES are sites only in
        the two-level models, so before `theta_raw` was kept, opting into session sites
        saved nothing a latent decision-variable replay could use. Asserted on the
        artifact's own `sites_saved` rather than on the tuple, because `save_fit` filters
        the keep list against the sites the sampler actually produced -- which is exactly
        how the five names came to be a silent no-op here.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with_sessions = save_fit(
                self.mcmc, tmp, name="c", include_session_sites=True
            )
            saved = json.loads(Path(with_sessions["json"]).read_text())["sites_saved"]
            self.assertIn("theta_raw", saved)

            # Reconstructible offline: theta_raw plus the subject-level sites this fit
            # already keeps. Shapes must line up as (draws, subjects, sessions, params)
            # against (draws, subjects, params), or the broadcast below is wrong.
            draws = self.mcmc.get_samples()
            theta_raw = np.asarray(draws["theta_raw"])
            mu_p = np.asarray(draws["mu_p"])
            self.assertEqual(theta_raw.ndim, 4)
            self.assertEqual(theta_raw.shape[0], mu_p.shape[0])
            self.assertEqual(theta_raw.shape[1], mu_p.shape[1])
            self.assertEqual(theta_raw.shape[3], mu_p.shape[2])

    def test_population_round_trips(self):
        """Draws read back match the sampler's, so a rescore needs no refit."""
        with tempfile.TemporaryDirectory() as tmp:
            result = save_fit(self.mcmc, tmp, name="t")
            loaded = load_population(result["netcdf"])
            original = np.asarray(self.mcmc.get_samples()["population_mean"])
            self.assertEqual(loaded["population_mean"].shape, original.shape)
            np.testing.assert_allclose(
                loaded["population_mean"], original, rtol=1e-5, atol=1e-6
            )

    def test_arviz_can_reload_the_fit(self):
        """`az.from_netcdf` returns a populated InferenceData, not an empty one.

        Regression test. `save_fit` used to hand each group to `to_netcdf` as a bare
        Dataset, which writes the variables at the netCDF root. `az.from_netcdf` -- the
        documented way to reload a fit, and what any figure or rescoring code reaches for
        -- then found no groups and returned an EMPTY InferenceData *without raising*.
        The draws were intact but unreachable by the standard call, so the failure was
        silent.
        """
        import arviz as az

        with tempfile.TemporaryDirectory() as tmp:
            result = save_fit(self.mcmc, tmp, name="t")
            idata = az.from_netcdf(result["netcdf"])

            # arviz is pinned to >=1.0,<2, where from_netcdf returns an xarray DataTree
            # and `.groups` is a property of node paths ("/posterior"). Written for that
            # line only, on purpose: the pin exists so this file does not have to carry
            # branches for an arviz nobody installs.
            groups = [g.lstrip("/") for g in idata.groups if g != "/"]
            self.assertIn("posterior", groups, f"no posterior group; groups={groups}")
            self.assertIn("sample_stats", groups, f"no sample_stats group; groups={groups}")

            posterior = idata["posterior"]
            for site in POPULATION_SITES:
                self.assertIn(site, posterior)
            original = np.asarray(self.mcmc.get_samples()["population_mean"])
            np.testing.assert_allclose(
                np.asarray(posterior["population_mean"]).reshape(-1, original.shape[-1]),
                original,
                rtol=1e-5, atol=1e-6,
            )
            # Diagnostics must travel in the same object, or r_hat/ESS cannot be
            # computed from the artifact the way arviz expects.
            self.assertIn("diverging", idata["sample_stats"])

    def test_records_provenance(self):
        """Supplied metadata and the source commit are recorded."""
        with tempfile.TemporaryDirectory() as tmp:
            result = save_fit(
                self.mcmc, tmp, name="t",
                meta={"seed": 7, "subject_ratio": 0.049, "snapshot": "20260603"},
            )
            meta = json.loads(Path(result["json"]).read_text())["_meta"]
            self.assertEqual(meta["seed"], 7)
            self.assertEqual(meta["snapshot"], "20260603")
            self.assertIn("models_git_sha", meta)

    def test_diagnostics_cover_convergence(self):
        """Diagnostics report the numbers a fit should be judged on."""
        diagnostics = summarise_diagnostics(
            __import__("arviz").from_numpyro(self.mcmc)
        )
        self.assertIn("divergences", diagnostics)
        for key in ("max_r_hat", "min_ess_bulk"):
            self.assertIn(key, diagnostics)


if __name__ == "__main__":
    unittest.main()
