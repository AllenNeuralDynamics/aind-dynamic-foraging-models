"""Cross-check against Linderman's ``ssm`` (as Faeze used it; Ashwood et al. used its C = 2 form).

The reference emission is ``ssm``'s ``InputDrivenObservations`` with ``C = 3`` and sticky
transitions, so both implementations can fit the same model to the same data from the same
start under the same priors (Gaussian sigma = 100 on the weights, Dirichlet alpha = 1, MLE
initial state, stopping at 1e-6 nats per trial). Three steps, because ``ssm`` needs numpy < 2
and a C compiler, so it lives in its own environment:

    python ssm_parity.py ours  results/ssm_parity_shared.npz    # [glmhmm] environment
    python ssm_parity.py ssm   results/ssm_parity_shared.npz    # an environment with ssm
    python ssm_parity.py compare results/ssm_parity_shared.npz  # [glmhmm] environment

``compare`` writes ``results/ssm_parity.json``; ``plot_recovery.py`` tables it in RESULTS.md.
The shared ``.npz`` files are intermediate and not committed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K, N_SESSIONS, SEED = 3, 100, 2026
SIGMA, ALPHA, TOL_PER_TRIAL = 100.0, 1.0, 1e-6


def _sessions(d):
    """(choices, rewards) per session from a shared file."""
    splits = np.cumsum(d["lengths"])[:-1]
    return list(zip(np.split(d["choices"], splits), np.split(d["rewards"], splits)))


def _pack(prefix, params):
    """Model parameters as flat arrays keyed for the shared file."""
    return {f"{prefix}_pi": np.asarray(params.initial.probs),
            f"{prefix}_A": np.asarray(params.transitions.transition_matrix),
            f"{prefix}_W": np.asarray(params.emissions.weights)}


def ours(shared):
    """Simulate from the reference truth, make a shared start, and fit it with this package."""
    import jax
    from jax.experimental import enable_x64

    from aind_dynamic_foraging_models.glm_hmm import GLMHMM, Prior, fit_em, make_buckets
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs, column_slices
    from parameter_recovery import N_LAGS, TRUTH, simulate, truth_params

    truth_file = json.loads(TRUTH.read_text())
    rng = np.random.default_rng(SEED)
    lengths = rng.choice(np.asarray(truth_file["source"]["session_lengths"]), N_SESSIONS)
    sessions, _ = simulate("reference", truth_file["reference"], lengths, rng)
    prior = Prior(SIGMA, ALPHA, 0.0)
    n_trials = sum(len(c) for c, _ in sessions)
    with enable_x64():
        model = GLMHMM(K, N_LAGS, "reference")
        buckets = make_buckets(sessions, N_LAGS)
        truth, props = truth_params(model, truth_file["reference"])
        single = GLMHMM(1, N_LAGS, "reference")
        base = fit_em(single, *single.initialize(), buckets, prior)
        start, _ = model.initialize(jax.random.PRNGKey(0), noise_scale=0.2,
                                    weights=np.asarray(base.params.emissions.weights)[0])
        started = time.time()
        fit = fit_em(model, start, props, buckets, prior, max_iters=300, tol=TOL_PER_TRIAL)
        seconds = time.time() - started
    cols = column_slices(N_LAGS)
    X = np.concatenate([build_inputs(c, r, N_LAGS)[:, cols["regressors"]] for c, r in sessions])
    np.savez(shared, lengths=[len(c) for c, _ in sessions], n_trials=n_trials,
             choices=np.concatenate([c for c, _ in sessions]),
             rewards=np.concatenate([r for _, r in sessions]), X=X,
             ours_iters=len(fit.log_joint) - 1, ours_seconds=seconds,
             **_pack("truth", truth), **_pack("start", start), **_pack("ours", fit.params))
    print(f"ours: {n_trials} trials, {len(fit.log_joint) - 1} EM iterations, {seconds:.1f}s")


def run_ssm(shared):
    """Score the shared parameter sets with ssm, then fit ssm from the shared start."""
    import ssm

    d = dict(np.load(shared))
    splits = np.cumsum(d["lengths"])[:-1]
    datas = [c[:, None].astype(int) for c in np.split(d["choices"], splits)]
    inputs = list(np.split(d["X"], splits))
    hmm = ssm.HMM(K, 1, d["X"].shape[1], observations="input_driven_obs",
                  observation_kwargs=dict(C=3, prior_sigma=SIGMA), transitions="sticky",
                  transition_kwargs=dict(alpha=ALPHA, kappa=0.0))

    def load(prefix):
        """Set ssm's parameters from the shared file."""
        hmm.init_state_distn.log_pi0 = np.log(d[f"{prefix}_pi"])
        hmm.transitions.log_Ps = np.log(d[f"{prefix}_A"])
        hmm.observations.Wk = d[f"{prefix}_W"].copy()

    for prefix in ("truth", "start", "ours"):
        load(prefix)
        d[f"ssm_ll_at_{prefix}"] = hmm.log_likelihood(datas, inputs=inputs)
    load("start")
    started = time.time()
    lls = hmm.fit(datas, inputs=inputs, method="em", num_iters=300, initialize=False,
                  tolerance=TOL_PER_TRIAL * int(d["n_trials"]), verbose=0)
    d.update(ssm_seconds=time.time() - started, ssm_iters=len(lls) - 1,
             ssm_pi=np.exp(hmm.init_state_distn.log_pi0), ssm_A=np.exp(hmm.transitions.log_Ps),
             ssm_W=hmm.observations.Wk.copy())
    np.savez(shared, **d)
    print(f"ssm: {d['ssm_iters']} EM iterations, {d['ssm_seconds']:.1f}s")


def compare(shared):
    """Both implementations' likelihoods of every parameter set, and how the fits differ."""
    from jax.experimental import enable_x64

    from aind_dynamic_foraging_models.glm_hmm import (
        GLMHMM,
        Prior,
        fit_em,
        make_buckets,
        match_by_weights,
        predictive_log_probs,
    )
    from parameter_recovery import N_LAGS

    d = np.load(shared)
    sessions = _sessions(d)
    model = GLMHMM(K, N_LAGS, "reference")
    prior = Prior(SIGMA, ALPHA, 0.0)
    out = {"n_trials": int(d["n_trials"]), "n_sessions": len(sessions),
           "ours_iters": int(d["ours_iters"]), "ours_seconds": float(d["ours_seconds"]),
           "ssm_iters": int(d["ssm_iters"]), "ssm_seconds": float(d["ssm_seconds"])}
    with enable_x64():
        buckets = make_buckets(sessions, N_LAGS)
        params = {}
        for name in ("truth", "start", "ours", "ssm"):
            params[name], props = model.initialize(initial_probs=d[f"{name}_pi"],
                                                   transition_matrix=d[f"{name}_A"],
                                                   weights=d[f"{name}_W"])
            joint = fit_em(model, params[name], props, buckets, prior, max_iters=0)
            out[f"ours_ll_at_{name}"] = float(joint.log_joint[0]
                                              - model.log_prior(params[name], prior))
        pred = {k: np.concatenate([np.exp(v[i]) for i in range(len(sessions))])
                for k, v in ((n, predictive_log_probs(model, params[n], buckets))
                             for n in ("truth", "ours", "ssm"))}
    for name in ("truth", "start", "ours"):
        out[f"ssm_ll_at_{name}"] = float(d[f"ssm_ll_at_{name}"])
    for a, b in (("ours", "truth"), ("ssm", "truth"), ("ssm", "ours")):
        out[f"predictive_mae_{a}_vs_{b}"] = float(np.abs(pred[a] - pred[b]).mean())

    def side(W):
        """Side contrast (right - left) per state."""
        return (W[:, 1] - W[:, 0]).reshape(len(W), -1)

    for name in ("ours", "ssm"):
        perm, _ = match_by_weights(side(d[f"{name}_W"]), side(d["truth_W"]))
        out[f"{name}_side_rmse_to_truth_by_state"] = np.sqrt(
            ((side(d[f"{name}_W"])[perm] - side(d["truth_W"])) ** 2).mean(1)).tolist()
        out[f"{name}_transition_max_err"] = float(np.abs(
            d[f"{name}_A"][np.ix_(perm, perm)] - d["truth_A"]).max())
    perm, _ = match_by_weights(side(d["ssm_W"]), side(d["ours_W"]))
    out["ssm_vs_ours_side_rmse_by_state"] = np.sqrt(
        ((side(d["ssm_W"])[perm] - side(d["ours_W"])) ** 2).mean(1)).tolist()
    (HERE / "results" / "ssm_parity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    step, shared = sys.argv[1], Path(sys.argv[2])
    {"ours": ours, "ssm": run_ssm, "compare": compare}[step](shared)
