"""Two-stage empirical Bayes: the population stage and the full driver."""

import unittest

import numpy as np

try:
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes.two_stage import (
        POPULATION_COORDS,
        fit_population,
        summarise_subject_posterior,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestPopulationStage(unittest.TestCase):
    """The random-effects population model over subject-level posteriors."""

    def test_separates_between_subject_spread_from_posterior_noise(self):
        """Between-subject scale is recovered net of within-subject uncertainty.

        This is the failure mode the two-stage fit exists to avoid: feeding the population
        stage bare posterior means, as if they were exact, conflates each subject's
        posterior uncertainty with genuine between-subject variance and inflates the
        population scale -- which would hand held-out subjects a too-diffuse prior.
        """
        rng = np.random.default_rng(0)
        n_subjects, n_coords = 40, len(POPULATION_COORDS)

        true_scale = 0.3
        measurement_error = 0.4  # deliberately larger than the real spread
        subject_true = 0.0 + true_scale * rng.standard_normal((n_subjects, n_coords))
        estimates = subject_true + measurement_error * rng.standard_normal((n_subjects, n_coords))
        standard_errors = np.full((n_subjects, n_coords), measurement_error)

        honest = fit_population(
            estimates, standard_errors, rng_key=jax.random.PRNGKey(0),
            num_warmup=400, num_samples=400,
        )
        honest_scale = float(np.mean(honest.get_samples()["population_scale"]))

        # Pretending the posterior means are exact observations.
        naive = fit_population(
            estimates, np.full_like(standard_errors, 1e-3), rng_key=jax.random.PRNGKey(0),
            num_warmup=400, num_samples=400,
        )
        naive_scale = float(np.mean(naive.get_samples()["population_scale"]))

        # The honest fit lands near the truth; the naive one inflates toward
        # sqrt(true_scale**2 + measurement_error**2) == 0.5.
        self.assertAlmostEqual(honest_scale, true_scale, delta=0.12)
        self.assertGreater(naive_scale, honest_scale + 0.1)
        self.assertAlmostEqual(naive_scale, np.hypot(true_scale, measurement_error), delta=0.12)

    def test_recovers_population_mean(self):
        """The population location is recovered from noisy subject estimates."""
        rng = np.random.default_rng(1)
        n_subjects, n_coords = 40, len(POPULATION_COORDS)
        true_mean = np.linspace(-0.5, 0.5, n_coords)

        subject_true = true_mean + 0.3 * rng.standard_normal((n_subjects, n_coords))
        standard_errors = np.full((n_subjects, n_coords), 0.2)
        estimates = subject_true + standard_errors * rng.standard_normal((n_subjects, n_coords))

        mcmc = fit_population(
            estimates, standard_errors, rng_key=jax.random.PRNGKey(0),
            num_warmup=400, num_samples=400,
        )
        posterior_mean = np.asarray(mcmc.get_samples()["population_mean"]).mean(axis=0)
        np.testing.assert_allclose(posterior_mean, true_mean, atol=0.2)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestSubjectSummary(unittest.TestCase):
    """Reducing a subject's posterior draws to population-stage coordinates."""

    def test_stacks_location_then_log_scale(self):
        """Coordinates are mu_p for every parameter, then log_sigma for every parameter."""
        n_draws, n_params = 50, len(POPULATION_COORDS) // 2
        samples = {
            "mu_p": np.tile(np.arange(n_params, dtype=float), (n_draws, 1)),
            "log_sigma": np.tile(np.arange(n_params, dtype=float) + 100.0, (n_draws, 1)),
        }
        mean, standard_error = summarise_subject_posterior(samples)

        self.assertEqual(mean.shape, (len(POPULATION_COORDS),))
        self.assertEqual(standard_error.shape, (len(POPULATION_COORDS),))
        np.testing.assert_allclose(mean[:n_params], np.arange(n_params))
        np.testing.assert_allclose(mean[n_params:], np.arange(n_params) + 100.0)
        # Constant draws carry no uncertainty.
        np.testing.assert_allclose(standard_error, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
