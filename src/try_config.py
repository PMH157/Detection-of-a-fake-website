"""Sandbox: evaluate a hand-picked RF configuration against the final decision.

Does NOT touch results/, figures/, README or the report. Output goes to results_trial/.
Example:  python src/try_config.py --n-estimators 50 --max-depth 15
"""
import argparse
import json
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from experiment import RESULTS, SPLIT_SEED, TEST_SIZE, cv_evaluate, load_data, test_evaluate
from step_summary import to_kwargs


def chosen_config():
    """The final decision (results/decision.json, written by step_summary.py)."""
    return to_kwargs(json.loads((RESULTS / "decision.json").read_text())["params"])


def main():
    base = chosen_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--min-samples-leaf", type=int, default=base["min_samples_leaf"])
    ap.add_argument("--max-features", type=float, default=base["max_features"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS.parent / "results_trial"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

    trial = {"n_estimators": a.n_estimators, "max_depth": a.max_depth,
             "min_samples_leaf": a.min_samples_leaf, "max_features": a.max_features}
    name = f"trees{a.n_estimators}_depth{a.max_depth}_leaf{a.min_samples_leaf}_feat{a.max_features:.2f}"
    X, y, _ = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)

    rows = []
    for label, p in (("final decision (OFAT V10)", base), (f"trial {name}", {**base, **trial})):
        for seed in range(a.seeds):
            f1, _ = cv_evaluate(p, Xtr, ytr, seed)
            rows.append({"config": label, "seed": seed, **p, "cv_f1": f1,
                         **test_evaluate(p, Xtr, ytr, Xte, yte, seed)})
    df = pd.DataFrame(rows)
    cols = ["cv_f1", "test_f1", "test_precision", "test_recall", "size_mb", "latency_ms_per_1k", "fit_s"]
    summary = df.groupby("config", sort=False)[cols].agg(["mean", "std"])

    out = pd.io.common.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_runs.csv", index=False)
    summary.to_csv(out / f"{name}_summary.csv")
    print(f"{a.seeds} seeds | same split and CV as the main experiment | other hyperparameters = final decision")
    print(summary.round(4).T.to_string())
    print(f"\nwritten to {out}/{name}_runs.csv and _summary.csv")


if __name__ == "__main__":
    main()
