"""Constrained HPO for a Random Forest phishing-website detector.

Compares, under the same objective, data split and evaluation budget:
  - Default : sklearn RandomForestClassifier defaults (no search)
  - Random  : Random Search  (optuna.samplers.RandomSampler)
  - TPE     : Bayesian / TPE (optuna.samplers.TPESampler)

Decision variables x = (n_estimators, max_depth, min_samples_leaf, max_features)
Objective          J(x) = F1_cv(x) - LAMBDA * max(0, size_mb(x) - S_MAX) / S_MAX   (maximize)
Constraint         g(x) = size_mb(x) - S_MAX <= 0                                    (model size budget)
The test set is touched only once per run, after the search has picked x*.
"""
import argparse
import io
import json
import pickle
import time
import warnings
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
DATA_CSV = ROOT / "data" / "phishing.csv"
RESULTS = ROOT / "results"

SPLIT_SEED = 42      # fixed train/test split, shared by every method and seed
TEST_SIZE = 0.2
CV_FOLDS = 3
S_MAX = 1.0          # model-size budget in MB (e.g. shipping the model in a browser extension)
LAMBDA = 1.0         # penalty weight for violating the size budget

SPACE = {  # search space Omega
    "n_estimators": (10, 300),      # int, log scale
    "max_depth": (2, 30),           # int
    "min_samples_leaf": (1, 20),    # int
    "max_features": (0.05, 1.0),    # float, fraction of the 30 features
}


def load_data():
    """UCI Phishing Websites (id 327). Label 1 = phishing (Result == -1), 0 = legitimate."""
    if not DATA_CSV.exists():
        raw = urllib.request.urlopen(DATA_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            arff = z.read("Training Dataset.arff").decode()  # the zip also has a smaller .old.arff
        lines = arff.splitlines()
        names = [l.split()[1] for l in lines if l.lower().startswith("@attribute")]
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1
        rows = [l for l in lines[start:] if l.strip() and not l.startswith("%")]
        df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=names)
        DATA_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
    df = pd.read_csv(DATA_CSV)
    n_raw = len(df)
    # Exact duplicate rows would leak between train/CV/test folds and inflate scores.
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.iloc[:, :-1].to_numpy()
    y = (df.iloc[:, -1] == -1).astype(int).to_numpy()
    return X, y, n_raw


def penalized(f1, size_mb, s_max=S_MAX, lam=LAMBDA):
    """J(x): CV F1 minus a relative penalty when the size constraint is violated."""
    return f1 - lam * max(0.0, size_mb - s_max) / s_max


def model_size_mb(model):
    return len(pickle.dumps(model)) / 1e6


def cv_evaluate(params, X, y, seed):
    """Return mean CV F1 and the size (MB) of the model that would be deployed for x."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    f1s = []
    for tr, va in cv.split(X, y):
        m = RandomForestClassifier(**params, n_jobs=-1, random_state=seed).fit(X[tr], y[tr])
        f1s.append(f1_score(y[va], m.predict(X[va])))
    # Size must be measured on the full-train refit: fold models see 2/3 of the rows, grow
    # smaller trees, and underestimate the deployed size (first run: 30-40% of x* infeasible).
    full = RandomForestClassifier(**params, n_jobs=-1, random_state=seed).fit(X, y)
    return float(np.mean(f1s)), model_size_mb(full)


def suggest(trial):
    lo, hi = SPACE["n_estimators"]
    return {
        "n_estimators": trial.suggest_int("n_estimators", lo, hi, log=True),
        "max_depth": trial.suggest_int("max_depth", *SPACE["max_depth"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *SPACE["min_samples_leaf"]),
        "max_features": trial.suggest_float("max_features", *SPACE["max_features"]),
    }


def latency_ms_per_1k(model, X, repeats=5):
    """Median single-thread prediction time per 1000 samples."""
    model.set_params(n_jobs=1)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t)
    return float(np.median(times)) / len(X) * 1000 * 1000


def test_evaluate(params, Xtr, ytr, Xte, yte, seed):
    t = time.perf_counter()
    m = RandomForestClassifier(**params, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    p = m.predict(Xte)
    size = model_size_mb(m)
    return {
        "test_f1": f1_score(yte, p), "test_accuracy": accuracy_score(yte, p),
        "test_precision": precision_score(yte, p), "test_recall": recall_score(yte, p),
        "size_mb": size, "feasible": size <= S_MAX,
        "latency_ms_per_1k": latency_ms_per_1k(m, Xte), "fit_s": fit_s,
    }


def run_search(method, seed, n_trials, Xtr, ytr):
    sampler = (optuna.samplers.TPESampler(seed=seed, n_startup_trials=10) if method == "TPE"
               else optuna.samplers.RandomSampler(seed=seed))
    rows = []

    def objective(trial):
        params = suggest(trial)
        t = time.perf_counter()
        f1, size = cv_evaluate(params, Xtr, ytr, seed)
        j = penalized(f1, size)
        rows.append({"method": method, "seed": seed, "trial": trial.number, **params,
                     "cv_f1": f1, "size_mb": size, "J": j, "feasible": size <= S_MAX,
                     "eval_s": time.perf_counter() - t})
        return j

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of independent seeds per method")
    ap.add_argument("--trials", type=int, default=40, help="evaluation budget (trials) per search run")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # harmless sklearn/joblib version-mismatch warning emitted on every parallel predict
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, n_raw = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
    print(f"data: {n_raw} raw rows, {len(X)} after dedup | train {len(Xtr)} / test {len(Xte)} | phishing rate {y.mean():.3f}")

    trials, finals = [], []
    defaults = {k: getattr(RandomForestClassifier(), k) for k in SPACE}
    for seed in range(args.seeds):
        f1, size = cv_evaluate({}, Xtr, ytr, seed)
        trials.append({"method": "Default", "seed": seed, "trial": 0, **defaults, "cv_f1": f1,
                       "size_mb": size, "J": penalized(f1, size), "feasible": size <= S_MAX, "eval_s": np.nan})
        finals.append({"method": "Default", "seed": seed, "cv_f1": f1, "J": penalized(f1, size),
                       **defaults, **test_evaluate({}, Xtr, ytr, Xte, yte, seed)})
        for method in ("Random", "TPE"):
            t = time.perf_counter()
            best, rows = run_search(method, seed, args.trials, Xtr, ytr)
            search_s = time.perf_counter() - t
            trials += rows
            br = max(rows, key=lambda r: r["J"])
            finals.append({"method": method, "seed": seed, "cv_f1": br["cv_f1"], "J": br["J"],
                           "search_s": search_s, **best, **test_evaluate(best, Xtr, ytr, Xte, yte, seed)})
            print(f"seed {seed} {method:6s} J={br['J']:.4f} cv_f1={br['cv_f1']:.4f} size={br['size_mb']:.2f}MB ({search_s:.0f}s)")

    pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
    final = pd.DataFrame(finals)
    final.to_csv(out / "final.csv", index=False)
    cols = ["J", "cv_f1", "test_f1", "test_accuracy", "test_recall", "size_mb", "latency_ms_per_1k", "search_s"]
    summary = final.groupby("method", sort=False)[cols].agg(["mean", "std"])
    summary["feasible_rate"] = final.groupby("method", sort=False)["feasible"].mean()
    summary.to_csv(out / "summary.csv")
    (out / "config.json").write_text(json.dumps({"seeds": args.seeds, "trials": args.trials, "split_seed": SPLIT_SEED, "test_size": TEST_SIZE,
               "cv_folds": CV_FOLDS, "s_max_mb": S_MAX, "lambda": LAMBDA, "space": SPACE,
               "n_raw": n_raw, "n_dedup": int(len(X)), "sklearn_defaults": defaults},
              indent=2, default=str))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
