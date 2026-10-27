"""Parameter recovery for the GLM-HMM, at parameters fitted to real mice.

Ground truth (``truth_K3_L5.json``) is a K = 3, L = 5 global fit of each emission on real
sessions (snapshot 20260603, 10 training mice; provenance in the file). Data are simulated
closed-loop: the GLM-HMM chooses left / right / ignore from its own history on a coupled
baiting task with blocks, so the regressors have realistic statistics. Each data set is
refitted with the production protocol (K = 1 first, then K > 1 from its weights plus noise,
with restarts) and compared with the truth.

Experiments (``--experiment``):

``size``
    Recovery against data size (25, 100, 400 sessions; 3 seeds): weight and transition
    errors, class probabilities under the generating state, state decoding against the
    oracle that knows the true parameters, and the fit's log joint against the truth's.
``selection``
    K = 1..5 fitted to data from the K = 3 truth; held-out simulated sessions scored with
    the forward filter. The score should rise to K = 3 and stop.
``per_mouse``
    The headline protocol: 12 mice whose weights scatter around the truth, a global fit on
    every mouse's context sessions, then per-mouse MAP refits, scored on eval sessions. Each
    refit should move from the global fit toward that mouse's own truth.

Run from this directory (each experiment writes ``results/<experiment>_<emission>.json``)::

    python parameter_recovery.py --experiment size --emission hurdle
    python parameter_recovery.py --plot
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRUTH = HERE / "truth_K3_L5.json"
RESULTS = HERE / "results"
N_LAGS = 5
SIZES = (25, 100, 400)
SEEDS = (0, 1, 2)
RESTARTS = 5


# -- task and simulation --------------------------------------------------------------------


class CoupledBaitingTask:
    """Coupled block task with baiting, like the mice's ``Coupled Baiting`` curriculum.

    Reward probabilities sum to 0.45 and come in left:right ratios of 8, 6, 3 or 1 (either
    side richer), in blocks of 20-60 trials. A side that is baited stays baited until it is
    chosen; an ignore collects nothing and leaves the bait in place.
    """

    RATIOS = (8.0, 6.0, 3.0, 1.0)
    P_SUM = 0.45

    def __init__(self, rng):
        """Start the first block."""
        self.rng = rng
        self.baited = [False, False]
        self._new_block()

    def _new_block(self):
        """Draw a block's reward probabilities and length."""
        ratio = self.rng.choice(self.RATIOS)
        rich = self.P_SUM * ratio / (1.0 + ratio)
        probs = [rich, self.P_SUM - rich]
        self.probs = probs if self.rng.random() < 0.5 else probs[::-1]
        self.remaining = int(self.rng.integers(20, 61))

    def step(self, choice):
        """Bait, collect the reward of ``choice`` (0 left, 1 right, 2 ignore), advance."""
        for side in (0, 1):
            if not self.baited[side] and self.rng.random() < self.probs[side]:
                self.baited[side] = True
        reward = 0
        if choice in (0, 1) and self.baited[choice]:
            reward, self.baited[choice] = 1, False
        self.remaining -= 1
        if self.remaining == 0:
            self._new_block()
        return reward


def class_probs(emission, weights, x, state):
    """p(left, right, ignore | state, inputs row x), restated in NumPy from the model."""
    from aind_dynamic_foraging_models.glm_hmm.design import column_slices

    cols = column_slices(N_LAGS)
    if emission == "hurdle":
        side = np.append(x[cols["side"]], x[cols["bias"]]) @ weights["side_weights"][state]
        engage = np.append(x[cols["engage"]], x[cols["bias"]]) @ weights["engage_weights"][state]
        p_respond, p_right = 1 / (1 + np.exp(-engage)), 1 / (1 + np.exp(-side))
        return np.array([p_respond * (1 - p_right), p_respond * p_right, 1 - p_respond])
    logits = np.append(weights["weights"][state] @ x[cols["regressors"]], 0.0)
    e = np.exp(logits - logits.max())
    return e / e.sum()


def simulate(emission, truth, lengths, rng):
    """Closed-loop sessions: (list of (choices, rewards), list of true state paths)."""
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs

    weights = {k: np.asarray(v) for k, v in truth["emissions"].items()}
    A = np.asarray(truth["transition_matrix"])
    pi = np.asarray(truth["initial_probs"])
    sessions, paths = [], []
    for length in lengths:
        task, choices, rewards, path = CoupledBaitingTask(rng), [], [], []
        state = rng.choice(len(pi), p=pi)
        for t in range(length):
            if t:
                state = rng.choice(len(pi), p=A[state])
            recent = slice(max(0, t - N_LAGS), t)
            x = build_inputs(np.array(choices[recent] + [0]), np.array(rewards[recent] + [0]),
                             N_LAGS)[-1]
            choice = int(rng.choice(3, p=class_probs(emission, weights, x, state)))
            choices.append(choice)
            rewards.append(task.step(choice))
            path.append(state)
        sessions.append((np.array(choices), np.array(rewards)))
        paths.append(np.array(path))
    return sessions, paths


# -- fitting and scoring ----------------------------------------------------------------------


def truth_params(model, truth):
    """The truth as model parameters."""
    params, props = model.initialize(
        initial_probs=np.asarray(truth["initial_probs"]),
        transition_matrix=np.asarray(truth["transition_matrix"]),
        **{k: np.asarray(v) for k, v in truth["emissions"].items()},
    )
    return params, props


def fit_production(emission, num_states, buckets, seed, prior=None, restarts=RESTARTS):
    """K = 1 first, then K > 1 from its weights plus noise, keeping the best start."""
    import jax

    from aind_dynamic_foraging_models.glm_hmm import (
        GLMHMM,
        GLOBAL_PRIOR,
        fit_em,
        fit_with_restarts,
    )

    prior = GLOBAL_PRIOR if prior is None else prior
    single = GLMHMM(1, N_LAGS, emission)
    params, props = single.initialize()
    base = fit_em(single, params, props, buckets, prior)
    if num_states == 1:
        return single, base
    start = {k: np.asarray(v)[0] for k, v in base.params.emissions._asdict().items()}
    model = GLMHMM(num_states, N_LAGS, emission)
    best, _ = fit_with_restarts(model, buckets, jax.random.PRNGKey(seed), restarts, prior,
                                0.2, **start)
    return model, best


def flat_weights(params):
    """Every emission weight of each state, shape (K, P)."""
    leaves = [np.asarray(v).reshape(np.asarray(v).shape[0], -1)
              for v in params.emissions._asdict().values()]
    return np.concatenate(leaves, axis=1)


def align_to_paths(model, params, buckets, paths):
    """Permutation matching fitted states to true states, by posterior overlap with the paths."""
    from aind_dynamic_foraging_models.glm_hmm import match_by_posterior_overlap, state_posteriors

    post = state_posteriors(model, params, buckets)
    smoothed = np.concatenate([post[i]["smoothed"] for i in range(len(paths))])
    onehot = np.eye(model.num_states)[np.concatenate(paths)]
    permutation, quality, _ = match_by_posterior_overlap(smoothed, onehot)
    return permutation, quality, smoothed


def per_trial_loglik(model, params, buckets, sessions):
    """Summed one-step predictive log likelihood and trial count."""
    from aind_dynamic_foraging_models.glm_hmm import predictive_log_probs

    pred = predictive_log_probs(model, params, buckets)
    total = sum(pred[i][np.arange(len(c)), c].sum() for i, (c, _) in enumerate(sessions))
    return float(total), int(sum(len(c) for c, _ in sessions))


# -- experiments ------------------------------------------------------------------------------


def experiment_size(emission, truth, lengths_pool, sizes=SIZES, seeds=SEEDS):
    """Recovery at 25, 100 and 400 sessions, three seeds each."""
    from aind_dynamic_foraging_models.glm_hmm import (
        GLMHMM,
        GLOBAL_PRIOR,
        fit_em,
        make_buckets,
        state_posteriors,
    )

    rows = []
    true_model = GLMHMM(3, N_LAGS, emission)
    t_params, t_props = truth_params(true_model, truth)
    true_w = flat_weights(t_params)
    true_A = np.asarray(truth["transition_matrix"])
    for n_sessions in sizes:
        for seed in seeds:
            rng = np.random.default_rng(1000 * n_sessions + seed)
            lengths = rng.choice(lengths_pool, n_sessions)
            sessions, paths = simulate(emission, truth, lengths, rng)
            buckets = make_buckets(sessions, N_LAGS)
            started = time.time()
            model, fit = fit_production(emission, 3, buckets, seed)
            seconds = time.time() - started
            perm, quality, smoothed = align_to_paths(model, fit.params, buckets, paths)
            fit_w = flat_weights(fit.params)[perm]
            fit_A = np.asarray(fit.params.transitions.transition_matrix)[np.ix_(perm, perm)]
            states = np.concatenate(paths)
            inputs = np.vstack(_inputs(sessions))
            p_fit = np.exp(np.asarray(model.emission_component.class_log_probs(
                fit.params.emissions, inputs)))[:, perm]
            p_true = np.exp(np.asarray(true_model.emission_component.class_log_probs(
                t_params.emissions, inputs)))
            trials = np.arange(len(states))
            prob_err = np.abs(p_fit[trials, states] - p_true[trials, states])
            oracle = state_posteriors(true_model, t_params, buckets)
            oracle_smoothed = np.concatenate([oracle[i]["smoothed"] for i in range(len(paths))])
            at_truth = fit_em(true_model, t_params, t_props, buckets, GLOBAL_PRIOR,
                              max_iters=0).log_joint[0]
            rows.append({
                "n_sessions": int(n_sessions), "seed": int(seed), "n_trials": int(len(states)),
                "fit_seconds": seconds, "converged": bool(fit.converged),
                "weight_rmse": float(np.sqrt(np.mean((fit_w - true_w) ** 2))),
                "weight_rmse_by_state": np.sqrt(np.mean((fit_w - true_w) ** 2, 1)).tolist(),
                "transition_max_abs_err": float(np.max(np.abs(fit_A - true_A))),
                "class_prob_mae": float(prob_err.mean()),
                "class_prob_q99": float(np.quantile(prob_err, 0.99)),
                "decoding_accuracy": float(np.mean(smoothed[:, perm].argmax(1) == states)),
                "oracle_decoding_accuracy": float(np.mean(oracle_smoothed.argmax(1) == states)),
                "alignment_quality": quality,
                "log_joint_minus_truth": float(fit.log_joint[-1] - at_truth),
                "true_occupancy": np.bincount(states, minlength=3).tolist(),
                "fit_weights": fit_w.tolist(), "true_weights": true_w.tolist(),
                "fit_transitions": fit_A.tolist(),
            })
            print(f"{emission} n={n_sessions} seed={seed}: rmse {rows[-1]['weight_rmse']:.3f} "
                  f"A err {rows[-1]['transition_max_abs_err']:.4f} prob MAE "
                  f"{rows[-1]['class_prob_mae']:.4f} decode {rows[-1]['decoding_accuracy']:.3f}"
                  f" (oracle {rows[-1]['oracle_decoding_accuracy']:.3f}) {seconds:.0f}s",
                  flush=True)
    return {"rows": rows, "true_transitions": true_A.tolist()}


def _inputs(sessions):
    """Per-session inputs rows, in session order."""
    from aind_dynamic_foraging_models.glm_hmm.design import build_inputs

    return [build_inputs(c, r, N_LAGS) for c, r in sessions]


def experiment_selection(emission, truth, lengths_pool, n_train=100, n_test=50):
    """Held-out score against K for data from the K = 3 truth."""
    from aind_dynamic_foraging_models.glm_hmm import GLMHMM, make_buckets

    rng = np.random.default_rng(7)
    train, _ = simulate(emission, truth, rng.choice(lengths_pool, n_train), rng)
    test, _ = simulate(emission, truth, rng.choice(lengths_pool, n_test), rng)
    train_b, test_b = make_buckets(train, N_LAGS), make_buckets(test, N_LAGS)
    oracle_model = GLMHMM(3, N_LAGS, emission)
    oracle_ll, n = per_trial_loglik(oracle_model, truth_params(oracle_model, truth)[0], test_b,
                                    test)
    rows = []
    for k in range(1, 6):
        model, fit = fit_production(emission, k, train_b, seed=k)
        ll, n = per_trial_loglik(model, fit.params, test_b, test)
        rows.append({"num_states": k, "heldout_log_lik": ll, "n_trials": n,
                     "heldout_likelihood": float(np.exp(ll / n)),
                     "train_log_joint": float(fit.log_joint[-1])})
        print(f"{emission} K={k}: held-out {rows[-1]['heldout_likelihood']:.5f}", flush=True)
    return {"rows": rows, "oracle_likelihood": float(np.exp(oracle_ll / n)), "true_k": 3}


def experiment_per_mouse(emission, truth, lengths_pool, n_mice=12, n_sessions=28, spread=0.3):
    """Global fit on context sessions, per-mouse MAP refits, scored on eval sessions.

    28 sessions per mouse gives 14 context sessions, the median of the real held-out cohort.
    Each mouse's weights are the truth plus N(0, spread^2) on every coefficient: an assumed
    between-mouse spread, not a measured one.
    """
    from aind_dynamic_foraging_models.glm_hmm import SUBJECT_PRIOR, fit_em, make_buckets

    rng = np.random.default_rng(11)
    mice = []
    for mouse in range(n_mice):
        own = {k: (np.asarray(v) + spread * rng.standard_normal(np.asarray(v).shape)).tolist()
               for k, v in truth["emissions"].items()}
        mouse_truth = {**truth, "emissions": own}
        sessions, paths = simulate(emission, mouse_truth, rng.choice(lengths_pool, n_sessions),
                                   rng)
        mice.append({"truth": mouse_truth, "context": sessions[0::2], "context_paths": paths[0::2],
                     "eval": sessions[1::2]})
    context = [s for m in mice for s in m["context"]]
    context_paths = [p for m in mice for p in m["context_paths"]]
    context_b = make_buckets(context, N_LAGS)
    model, global_fit = fit_production(emission, 3, context_b, seed=0)
    perm, _, _ = align_to_paths(model, global_fit.params, context_b, context_paths)
    _, props = model.initialize()
    rows = []
    for index, m in enumerate(mice):
        cb, eb = make_buckets(m["context"], N_LAGS), make_buckets(m["eval"], N_LAGS)
        refit = fit_em(model, global_fit.params, props, cb, SUBJECT_PRIOR)
        own_params = truth_params(model, m["truth"])[0]
        own_w = flat_weights(own_params)
        # Each fit matched to this mouse's own true state paths, so a refit that relabelled
        # its states is not scored as a weight error.
        perm_global, quality_global, _ = align_to_paths(model, global_fit.params, cb,
                                                        m["context_paths"])
        perm_refit, quality_refit, _ = align_to_paths(model, refit.params, cb,
                                                      m["context_paths"])
        zero_ll, n = per_trial_loglik(model, global_fit.params, eb, m["eval"])
        refit_ll, _ = per_trial_loglik(model, refit.params, eb, m["eval"])
        oracle_ll, _ = per_trial_loglik(model, own_params, eb, m["eval"])
        rows.append({
            "mouse": index, "n_eval_trials": n,
            "global_weight_rmse": float(np.sqrt(np.mean(
                (flat_weights(global_fit.params)[perm_global] - own_w) ** 2))),
            "refit_weight_rmse": float(np.sqrt(np.mean(
                (flat_weights(refit.params)[perm_refit] - own_w) ** 2))),
            "global_weight_median_err": float(np.median(np.abs(
                flat_weights(global_fit.params)[perm_global] - own_w))),
            "refit_weight_median_err": float(np.median(np.abs(
                flat_weights(refit.params)[perm_refit] - own_w))),
            "refit_relabelled": bool(np.any(perm_refit != perm)),
            "global_weight_rmse_by_state": np.sqrt(np.mean(
                (flat_weights(global_fit.params)[perm_global] - own_w) ** 2, 1)).tolist(),
            "refit_weight_rmse_by_state": np.sqrt(np.mean(
                (flat_weights(refit.params)[perm_refit] - own_w) ** 2, 1)).tolist(),
            "context_occupancy": np.bincount(np.concatenate(m["context_paths"]),
                                             minlength=model.num_states).tolist(),
            "alignment_quality_global": quality_global, "alignment_quality_refit": quality_refit,
            "refit_em_iters": int(len(refit.log_joint) - 1),
            "zero_shot_likelihood": float(np.exp(zero_ll / n)),
            "refit_likelihood": float(np.exp(refit_ll / n)),
            "oracle_likelihood": float(np.exp(oracle_ll / n)),
        })
        print(f"{emission} mouse {index}: rmse global {rows[-1]['global_weight_rmse']:.3f} -> "
              f"refit {rows[-1]['refit_weight_rmse']:.3f} (median err "
              f"{rows[-1]['global_weight_median_err']:.3f} -> "
              f"{rows[-1]['refit_weight_median_err']:.3f}, relabelled "
              f"{rows[-1]['refit_relabelled']}); LL zero-shot "
              f"{rows[-1]['zero_shot_likelihood']:.4f} refit {rows[-1]['refit_likelihood']:.4f} "
              f"oracle {rows[-1]['oracle_likelihood']:.4f}", flush=True)
    return {"rows": rows, "spread": spread, "n_sessions": n_sessions}


EXPERIMENTS = {"size": experiment_size, "selection": experiment_selection,
               "per_mouse": experiment_per_mouse}


def main():
    """Run one experiment for one emission, or draw the figures from saved results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--emission", choices=("hurdle", "reference"))
    parser.add_argument("--plot", action="store_true", help="draw figures from results/")
    parser.add_argument("--quick", action="store_true",
                        help="size experiment at 10 sessions, one seed (a smoke test)")
    args = parser.parse_args()
    if args.plot:
        from plot_recovery import plot_all

        plot_all(RESULTS, HERE)
        return
    from jax.experimental import enable_x64

    truth_file = json.loads(TRUTH.read_text())
    lengths_pool = np.asarray(truth_file["source"]["session_lengths"])
    started = time.time()
    kwargs = {"sizes": (10,), "seeds": (0,)} if args.quick else {}
    with enable_x64():
        result = EXPERIMENTS[args.experiment](args.emission, truth_file[args.emission],
                                              lengths_pool, **kwargs)
    result.update({"experiment": args.experiment, "emission": args.emission,
                   "seconds": time.time() - started})
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.experiment}_{args.emission}{'_quick' if args.quick else ''}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out} in {result['seconds']:.0f}s")


if __name__ == "__main__":
    main()
