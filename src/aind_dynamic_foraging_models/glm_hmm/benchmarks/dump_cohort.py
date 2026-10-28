"""Write a cohort's train and eval sessions to an ``.npz`` for ``time_fit.py``.

    python dump_cohort.py --wrapper ../aind-dynamic-foraging-bfm-wrapper \\
        --dispatcher ../aind-dynamic-foraging-bfm-dispatcher --out cohort.npz [OVERRIDE ...]

This loads data through the wrapper's data loader and the dispatcher's Hydra configs, so it
needs both checkouts and a wrapper environment. The default overrides are study 14's
(``mice_snapshot_scaling``, snapshot 20260603, ``ignore_policy=include``); pass others to dump
another cohort. Real mouse data: keep the file out of the repository.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

STUDY_14 = [
    "data=mice_snapshot_scaling",
    "data.ignore_policy=include",
    "+data.cols_to_retain=[trial,subject_id,ses_idx,animal_response,earned_reward,"
    "current_stage_actual]",
    "model=glmhmm",
    "seed=0",
]


def main():
    """Load the cohort and write one flat array per split and field."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--dispatcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", default=STUDY_14)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    sys.path.insert(0, str(args.wrapper.resolve() / "code"))
    from model_trainers.glmhmm_trainer import _sessions

    started = time.time()
    config_dir = str(args.dispatcher.resolve() / "code" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=args.overrides)
    bundle = instantiate(cfg.data).load()
    print(f"loaded in {time.time() - started:.0f}s", flush=True)
    out = {}
    for split in ("train", "eval"):
        records = _sessions(bundle.raw, list(bundle.metadata[f"{split}_session_ids"]))
        out[f"{split}_choices"] = np.concatenate([r["choices"] for r in records]).astype(np.int8)
        out[f"{split}_rewards"] = np.concatenate([r["rewards"] for r in records]).astype(np.int8)
        out[f"{split}_lengths"] = np.array([len(r["choices"]) for r in records])
        out[f"{split}_subjects"] = np.array([r["subject_id"] for r in records])
        print(f"{split}: {len(records)} sessions, {out[f'{split}_lengths'].sum()} trials, "
              f"{len(set(out[f'{split}_subjects']))} subjects", flush=True)
    np.savez(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
