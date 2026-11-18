"""Overfitting diagnostics: train/CV/test gaps, HPO optimism on fresh splits, learning curves, leakage checks.

Writes results/overfitting_{gaps,splits,learning_curve}.csv, results/overfitting_leakage.json, figures/learning_curve.png.
Run after experiment.py, ofat_search.py, ofat_extra.py and step_summary.py (reads results/final.csv, the OFAT
versions and results/decision.json).
"""
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

import plots
from experiment import DATA_CSV, RESULTS, SPLIT_SEED, TEST_SIZE, cv_evaluate, load_data, model_size_mb
from ofat_extra import EXTRA
from ofat_search import DEFAULT, ORDER, coordinate_search, grids
from step_summary import PARAMS, load as load_steps, norm, to_kwargs

FIG = RESULTS.parent / "figures"


def params_of(row):
    if row.method == "Default":
        return {}
    return {"n_estimators": int(row.n_estimators), "max_depth": int(float(row.max_depth)),
            "min_samples_leaf": int(row.min_samples_leaf), "max_features": float(row.max_features)}


def gaps(Xtr, ytr, final):
    """Train F1 vs CV F1 vs test F1 of every final model (10 seeds x Default/Random/TPE/OFAT V10)."""
    v10 = load_steps().query("vnum == 10")
    models = [(r.method, int(r.seed), params_of(r), r.cv_f1, r.test_f1) for r in final.itertuples()]
    models += [("OFAT (V10)", int(r.seed), to_kwargs({k: getattr(r, k) for k in PARAMS}), r.cv_f1, r.test_f1)
               for r in v10.itertuples()]
    rows = []
    for method, seed, params, cv_f1, test_f1 in models:
        m = RandomForestClassifier(**params, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        rows.append({"method": method, "seed": seed, "train_f1": f1_score(ytr, m.predict(Xtr)),
                     "cv_f1": cv_f1, "test_f1": test_f1})
    df = pd.DataFrame(rows)
    df["train_minus_cv"] = df.train_f1 - df.cv_f1
    df["cv_minus_test"] = df.cv_f1 - df.test_f1
    df.to_csv(RESULTS / "overfitting_gaps.csv", index=False)
    print(df.groupby("method", sort=False)[["train_f1", "cv_f1", "test_f1", "train_minus_cv", "cv_minus_test"]]
          .agg(["mean", "std"]).round(4).to_string())


def ofat_v10(Xtr, ytr, seed=0):
    """The decision procedure itself: coordinate search V0-V4 then V5-V10 (objective = CV F1)."""
    def evaluate(p):
        f1, size = cv_evaluate(p, Xtr, ytr, seed)
        return f1, {"cv_f1": f1, "size_mb": size}
    a, _ = coordinate_search(DEFAULT, grids(), ORDER, evaluate)
    start = {**{k: a[-1][k] for k in ORDER}, **{k: g[0] for k, g in EXTRA.items()}}
    b, _ = coordinate_search(start, EXTRA, list(EXTRA), evaluate)
    return {k: b[-1][k] for k in ORDER + list(EXTRA)}, b[-1]["cv_f1"]


def fresh_splits(X, y, split_seeds=(1, 2, 3, 4, 5)):
    """Repeat the whole pipeline on new train/test splits: is CV-of-the-winner an optimistic estimate of test?"""
    rows = []
    for s in split_seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=s)
        f1, _ = cv_evaluate({}, Xtr, ytr, 0)
        m = RandomForestClassifier(n_jobs=-1, random_state=0).fit(Xtr, ytr)
        rows.append({"split_seed": s, "method": "Default", "cv_f1": f1, "test_f1": f1_score(yte, m.predict(Xte)),
                     "size_mb": model_size_mb(m)})
        t = time.perf_counter()
        best, best_cv = ofat_v10(Xtr, ytr)
        m = RandomForestClassifier(**best, n_jobs=-1, random_state=0).fit(Xtr, ytr)
        rows.append({"split_seed": s, "method": "OFAT (V10)", "cv_f1": best_cv, "test_f1": f1_score(yte, m.predict(Xte)),
                     "size_mb": model_size_mb(m), **{k: norm(v) for k, v in best.items()}})
        print(f"split {s}: default test {rows[-2]['test_f1']:.4f} | OFAT cv {best_cv:.4f} test {rows[-1]['test_f1']:.4f} "
              f"({time.perf_counter() - t:.0f}s)")
    df = pd.DataFrame(rows)
    df["cv_minus_test"] = df.cv_f1 - df.test_f1
    df.to_csv(RESULTS / "overfitting_splits.csv", index=False)
    print(df.groupby("method")[["cv_f1", "test_f1", "cv_minus_test", "size_mb"]].agg(["mean", "std"]).round(4).to_string())


def learning_curves(Xtr, ytr, final):
    chosen = to_kwargs(json.loads((RESULTS / "decision.json").read_text())["params"])  # the final decision
    models = {"Default": {}, "Chosen (OFAT V10)": chosen}
    sizes = np.linspace(0.1, 1.0, 8)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, p in models.items():
        n, tr, va = learning_curve(RandomForestClassifier(**p, n_jobs=-1, random_state=0), Xtr, ytr, cv=cv,
                                   train_sizes=sizes, scoring="f1", n_jobs=1)
        for i in range(len(n)):
            rows.append({"model": name, "n_train": int(n[i]), "train_f1_mean": tr[i].mean(), "train_f1_std": tr[i].std(),
                         "val_f1_mean": va[i].mean(), "val_f1_std": va[i].std()})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "overfitting_learning_curve.csv", index=False)
    print(df.round(4).to_string(index=False))

    plt = plots.plt
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, (name, color) in zip(axes, (("Default", plots.COLORS["Default"]), ("Chosen (OFAT V10)", plots.COLORS["TPE"]))):
        d = df[df.model == name]
        for col, ls, label in (("train", "--", "train F1"), ("val", "-", "validation F1 (5-fold)")):
            ax.plot(d.n_train, d[f"{col}_f1_mean"], ls=ls, color=color, marker="o", ms=5, label=label)
            ax.fill_between(d.n_train, d[f"{col}_f1_mean"] - d[f"{col}_f1_std"], d[f"{col}_f1_mean"] + d[f"{col}_f1_std"],
                            color=color, alpha=0.15, lw=0)
        ax.set(title=name, xlabel="Training rows")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.png", dpi=150)


def leakage(Xtr, Xte, ytr, yte):
    """After dedup: exact feature vectors shared by train and test, and feature vectors with conflicting labels."""
    key = lambda A: [r.tobytes() for r in A]
    tr_keys, te_keys = key(Xtr), key(Xte)
    tr_set = set(tr_keys)
    shared = np.array([k in tr_set for k in te_keys])
    raw = pd.read_csv(DATA_CSV).drop_duplicates()
    feats = raw.columns[:-1]
    conflict = raw.groupby(list(feats))[raw.columns[-1]].nunique()
    out = {"test_rows": int(len(Xte)), "test_features_seen_in_train": int(shared.sum()),
           "test_features_seen_in_train_rate": float(shared.mean()),
           "feature_vectors_with_conflicting_labels": int((conflict > 1).sum()),
           "rows_in_conflicting_groups": int(raw.set_index(list(feats)).index.isin(conflict[conflict > 1].index).sum())}
    # Of the shared test rows: does the train copy carry the same label? (if not, it is label noise, not leakage)
    lab = dict(zip(tr_keys, ytr))
    same = [lab[k] == yv for k, yv in zip(te_keys, yte) if k in lab]
    out["shared_with_same_label"] = int(sum(same))
    (RESULTS / "overfitting_leakage.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
    X, y, _ = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
    final = pd.read_csv(RESULTS / "final.csv")
    leakage(Xtr, Xte, ytr, yte)
    gaps(Xtr, ytr, final)
    learning_curves(Xtr, ytr, final)
    fresh_splits(X, y)
