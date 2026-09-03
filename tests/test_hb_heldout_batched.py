"""Batched adaptation: does fitting many subjects at once match fitting them one by one?

Held-out subjects are independent given a frozen population, so batching them into one
sampler is a large speed win on a device where extra lanes are nearly free. The catch is
that one sampler adapts a single step size across every subject's block. This measures
whether that costs anything, rather than assuming either way.
"""

import time
import unittest

from tests._hb_deps import assert_deps_present

import numpy as np
from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from scipy.stats import norm

from aind_dynamic_foraging_models.generative_model import ForagerCollection

try:
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        auto_session_chunk,
        batched_choice_prob,
        batched_heldout_log_lik,
        fit_adaptation,
        fit_adaptation_batched,
        pointwise_log_predictive_density,
        posterior_predictive_choice_prob,
    )

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_JAX = False

# A broken extra must not report OK by skipping every test that touches it. With
# AIND_HB_REQUIRE_DEPS=1 -- which the CI job that installs [bayes] sets -- a failed
# import becomes an error here instead of a run of silent skips.
assert_deps_present(HAS_JAX)

BETA_MAX = 10.0
N_PARAMS = 5
POPULATION = {
    "population_mean": np.array([0.3, -0.6, -0.8, 0.2, 0.0]),
    "population_scale": np.array([0.4] * N_PARAMS),
    "log_sigma_mean": np.array([-1.4] * N_PARAMS),
    "log_sigma_spread": np.array([0.3] * N_PARAMS),
}


def _simulate_cohort(n_subjects, n_sessions, n_trials, seed=0):
    """Simulate held-out subjects drawn from the population."""
    rng = np.random.default_rng(seed)
    mu = POPULATION["population_mean"] + POPULATION["population_scale"] * rng.standard_normal(
        (n_subjects, N_PARAMS)
    )
    choices = np.zeros((n_subjects, n_sessions, n_trials), dtype=int)
    rewards = np.zeros((n_subjects, n_sessions, n_trials), dtype=float)
    for s in range(n_subjects):
        theta = mu[s] + 0.25 * rng.standard_normal((n_sessions, N_PARAMS))
        for j in range(n_sessions):
            tag = s * 100 + j
            f = ForagerCollection().get_preset_forager("Hattori2019", seed=tag)
            f.set_params(
                learn_rate_rew=float(norm.cdf(theta[j, 0])),
                learn_rate_unrew=float(norm.cdf(theta[j, 1])),
                forget_rate_unchosen=float(norm.cdf(theta[j, 2])),
                softmax_inverse_temperature=float(norm.cdf(theta[j, 3]) * BETA_MAX),
                biasL=float(theta[j, 4]),
            )
            f.perform(CoupledBlockTask(reward_baiting=True, num_trials=n_trials, seed=tag))
            choices[s, j] = f.get_choice_history()
            rewards[s, j] = f.get_reward_history()
    return choices, rewards


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestBatchedAdaptation(unittest.TestCase):
    """Batched against sequential adaptation on the same subjects."""

    @classmethod
    def setUpClass(cls):
        """Simulate a small cohort, then adapt it both ways."""
        cls.n_subjects, n_context, n_trials = 6, 4, 200
        cls.choices, cls.rewards = _simulate_cohort(cls.n_subjects, n_context + 1, n_trials)
        cls.context_c = cls.choices[:, :n_context]
        cls.context_r = cls.rewards[:, :n_context]
        cls.test_c = cls.choices[:, n_context]
        cls.test_r = cls.rewards[:, n_context]

        started = time.time()
        cls.batched = fit_adaptation_batched(
            cls.context_c, cls.context_r, POPULATION,
            rng_key=jax.random.PRNGKey(0), num_warmup=300, num_samples=300,
        )
        cls.batched_seconds = time.time() - started

        started = time.time()
        cls.sequential = []
        for s in range(cls.n_subjects):
            cls.sequential.append(fit_adaptation(
                cls.context_c[s], cls.context_r[s], POPULATION,
                rng_key=jax.random.PRNGKey(s), num_warmup=300, num_samples=300,
            ))
        cls.sequential_seconds = time.time() - started

    def test_shapes_carry_a_subject_axis(self):
        """Batched draws are indexed by subject."""
        mu_p = np.asarray(self.batched["mu_p"])
        self.assertEqual(mu_p.shape[1:], (self.n_subjects, N_PARAMS))

    def test_posteriors_agree_with_sequential(self):
        """Per-subject posterior means match sequential fitting.

        A shared step size across subject blocks is the approximation batching makes; this
        is what bounds its cost.
        """
        batched_mu = np.asarray(self.batched["mu_p"]).mean(axis=0)
        for s in range(self.n_subjects):
            sequential_mu = np.asarray(self.sequential[s]["mu_p"]).mean(axis=0)
            with self.subTest(subject=s):
                np.testing.assert_allclose(
                    batched_mu[s][:3], sequential_mu[:3], atol=0.35
                )

    def test_scores_agree_with_sequential(self):
        """Held-out likelihoods match, which is the quantity that actually gets reported."""
        batched_total, sequential_total, n_total = 0.0, 0.0, 0
        for s in range(self.n_subjects):
            key = jax.random.PRNGKey(100 + s)
            batched_prob = batched_choice_prob(
                self.batched, s, self.test_c[s], self.test_r[s], rng_key=key
            )
            sequential_prob = posterior_predictive_choice_prob(
                self.sequential[s], self.test_c[s], self.test_r[s], rng_key=key
            )
            b, n = pointwise_log_predictive_density(batched_prob, self.test_c[s])
            q, _ = pointwise_log_predictive_density(sequential_prob, self.test_c[s])
            batched_total += b
            sequential_total += q
            n_total += n

        batched_lik = float(np.exp(batched_total / n_total))
        sequential_lik = float(np.exp(sequential_total / n_total))
        self.assertAlmostEqual(batched_lik, sequential_lik, delta=0.01)

    def test_batching_is_faster(self):
        """One batched fit beats one fit per subject."""
        self.assertLess(self.batched_seconds, self.sequential_seconds)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestBatchedScoring(unittest.TestCase):
    """Scoring every held-out session in one vmapped pass."""

    @classmethod
    def setUpClass(cls):
        """Adapt a small cohort, then hold out several sessions per subject."""
        cls.n_subjects, n_context, cls.n_score, n_trials = 5, 3, 3, 180
        choices, rewards = _simulate_cohort(
            cls.n_subjects, n_context + cls.n_score, n_trials, seed=3
        )
        cls.context_c, cls.context_r = choices[:, :n_context], rewards[:, :n_context]
        cls.score_c, cls.score_r = choices[:, n_context:], rewards[:, n_context:]
        cls.samples = fit_adaptation_batched(
            cls.context_c, cls.context_r, POPULATION,
            rng_key=jax.random.PRNGKey(0), num_warmup=250, num_samples=250,
        )
        # flatten (subject, session) into a session list, as the trainer does
        cls.subject_indices = np.repeat(np.arange(cls.n_subjects), cls.n_score)
        cls.flat_c = cls.score_c.reshape(-1, n_trials)
        cls.flat_r = cls.score_r.reshape(-1, n_trials)

    def test_matches_per_session_scoring(self):
        """One vmapped pass agrees with scoring each session on its own."""
        key = jax.random.PRNGKey(11)
        batched_ll, batched_n = batched_heldout_log_lik(
            self.samples, self.subject_indices, self.flat_c, self.flat_r,
            rng_key=key, session_chunk=4,
        )

        per_session_total = 0.0
        for position in range(len(self.subject_indices)):
            subject = int(self.subject_indices[position])
            prob = batched_choice_prob(
                self.samples, subject, self.flat_c[position], self.flat_r[position],
                rng_key=key,
            )
            total, _ = pointwise_log_predictive_density(prob, self.flat_c[position])
            per_session_total += total

        # Fresh session latents are redrawn, so agreement is statistical, not exact.
        batched_lik = float(np.exp(batched_ll.sum() / batched_n.sum()))
        per_session_lik = float(np.exp(per_session_total / batched_n.sum()))
        self.assertAlmostEqual(batched_lik, per_session_lik, delta=0.01)

    def test_counts_respect_the_mask(self):
        """Masked trials are excluded from both the likelihood and the trial count."""
        mask = np.ones_like(self.flat_c, dtype=bool)
        mask[:, ::2] = False
        _, counts = batched_heldout_log_lik(
            self.samples, self.subject_indices, self.flat_c, self.flat_r,
            valid_mask=mask, rng_key=jax.random.PRNGKey(0), session_chunk=4,
        )
        np.testing.assert_array_equal(counts, mask.sum(axis=1))

    def test_chunking_does_not_change_the_result(self):
        """Chunk size caps memory without altering the answer."""
        key = jax.random.PRNGKey(5)
        small, _ = batched_heldout_log_lik(
            self.samples, self.subject_indices, self.flat_c, self.flat_r,
            rng_key=key, session_chunk=2,
        )
        large, _ = batched_heldout_log_lik(
            self.samples, self.subject_indices, self.flat_c, self.flat_r,
            rng_key=key, session_chunk=64,
        )
        # Different chunking redraws latents differently; totals stay close.
        self.assertAlmostEqual(float(small.sum()), float(large.sum()),
                               delta=abs(float(large.sum())) * 0.02)


@unittest.skipUnless(HAS_JAX, "requires the 'bayes' extra (jax, numpyro)")
class TestAutoSessionChunk(unittest.TestCase):
    """Sizing the scoring batch from the device rather than a constant."""

    def test_returns_a_usable_size_on_this_device(self):
        """Whatever the device, the chunk is a positive int within its bounds."""
        chunk = auto_session_chunk(n_trials=1238, n_draws=500)
        self.assertIsInstance(chunk, int)
        self.assertGreaterEqual(chunk, 8)
        self.assertLessEqual(chunk, 4096)

    def test_shrinks_as_the_working_set_grows(self):
        """More draws or longer sessions mean fewer sessions per pass.

        The working set scales with draws and trials as well as sessions, which is exactly
        why a fixed chunk size is wrong in both directions.
        """
        import jax

        if jax.devices()[0].platform == "cpu":
            self.skipTest("CPU returns a fixed small chunk by design")
        base = auto_session_chunk(n_trials=500, n_draws=100)
        self.assertLessEqual(auto_session_chunk(n_trials=500, n_draws=1000), base)
        self.assertLessEqual(auto_session_chunk(n_trials=5000, n_draws=100), base)

    def test_cpu_stays_small(self):
        """On CPU the chunk stays small, because batching there is counterproductive."""
        import jax

        if jax.devices()[0].platform != "cpu":
            self.skipTest("only meaningful on CPU")
        self.assertLessEqual(auto_session_chunk(n_trials=1238, n_draws=500), 64)

    def test_respects_explicit_bounds(self):
        """Callers can pin the range when they know better than the heuristic."""
        self.assertEqual(
            auto_session_chunk(n_trials=100, n_draws=1, floor=64, ceiling=64), 64
        )
