"""Chapter-4 add-on experiments: sensitivity, ablation, robustness, scalability, feature importance.

Reuses the objective/search from experiment.py; reads results/final.csv for the main (S_max = 1 MB) runs.
Run with --replot to only redraw figures from saved CSVs.
Writes results/{sensitivity,ablation,robustness,scalability,feature_importance}.csv and figures/*.png.
"""
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import plots  # shared figure style + colours
from experiment import (DATA_CSV, RESULTS, S_MAX, SPACE, SPLIT_SEED, TEST_SIZE, cv_evaluate, load_data,
                        model_size_mb, run_search, test_evaluate)

SEEDS, TRIALS = 10, 40
FIG = RESULTS.parent / "figures"  # not plots.FIG: plots reads sys.argv at import
P = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]


def best_params(row):
    return {"n_estimators": int(row.n_estimators), "max_depth": int(row.max_depth),
            "min_samples_leaf": int(row.min_samples_leaf), "max_features": float(row.max_features)}


def tpe_runs(Xtr, ytr, Xte, yte, s_max, lam, label, seeds=SEEDS):
    rows = []
    for seed in range(seeds):
        t = time.perf_counter()
        best, trials = run_search("TPE", seed, TRIALS, Xtr, ytr, s_max, lam)
        br = max(trials, key=lambda r: r["J"])
        res = test_evaluate(best, Xtr, ytr, Xte, yte, seed)
        res["feasible"] = res["size_mb"] <= s_max
        rows.append({"setting": label, "s_max_mb": s_max, "lambda": lam, "seed": seed, "cv_f1": br["cv_f1"],
                     "search_s": time.perf_counter() - t, **best, **res})
        print(f"  {label} seed {seed}: cv_f1={br['cv_f1']:.4f} size={res['size_mb']:.2f}MB")
    return rows


def plot_sensitivity(g, final):
    d0 = final[final.method == "Default"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9))
    for ax, col, label in ((axes[0], "test_f1", "Test F1"), (axes[1], "size_mb", "Model size (MB, log scale)")):
        ax.errorbar(g.index, g[(col, "mean")], yerr=g[(col, "std")], color=plots.COLORS["TPE"], marker="o",
                    ms=8, capsize=4, label="TPE (mean ± std, 10 seeds)")
        ax.axhline(d0[col].mean(), color=plots.COLORS["Default"], ls="--", lw=1.5, label="Default RF (no budget)")
        ax.set_xscale("log")
        ax.set_xticks(g.index, [f"{v:g}" for v in g.index])
        ax.minorticks_off()
        ax.set(xlabel="Size budget S_max (MB, log scale)", title=label)
    axes[1].plot(g.index, g.index, color=plots.MUTED, ls=":", lw=1.2, label="size = S_max")
    axes[1].set_yscale("log")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "sensitivity_smax.png", dpi=150)


def sensitivity(Xtr, ytr, Xte, yte, final):
    rows = []
    for s in (0.25, 0.5, 2.0):
        rows += tpe_runs(Xtr, ytr, Xte, yte, s, 1.0, f"S_max={s}")
    main = final[final.method == "TPE"].assign(setting=f"S_max={S_MAX}", s_max_mb=S_MAX, **{"lambda": 1.0})
    df = pd.concat([pd.DataFrame(rows), main[[c for c in rows[0] if c in main]]], ignore_index=True)
    df.to_csv(RESULTS / "sensitivity.csv", index=False)
    g = df.groupby("s_max_mb")[["cv_f1", "test_f1", "size_mb", "latency_ms_per_1k", "search_s"]].agg(["mean", "std"])
    g["feasible_rate"] = df.groupby("s_max_mb")["feasible"].mean()
    print(g.round(4).to_string())

    plot_sensitivity(g, final)


def ablation(Xtr, ytr, Xte, yte, final):
    rows = tpe_runs(Xtr, ytr, Xte, yte, S_MAX, 0.0, "TPE without size penalty (lambda=0)")
    # Keep duplicates: same protocol on the raw 11 055 rows
    raw = pd.read_csv(DATA_CSV)
    Xr, yr = raw.iloc[:, :-1].to_numpy(), (raw.iloc[:, -1] == -1).astype(int).to_numpy()
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, yr, test_size=TEST_SIZE, stratify=yr, random_state=SPLIT_SEED)
    seen = {r.tobytes() for r in np.c_[Xtr2, ytr2]}
    leak = np.mean([r.tobytes() in seen for r in np.c_[Xte2, yte2]])
    for seed in range(SEEDS):
        f1, size = cv_evaluate({}, Xtr2, ytr2, seed)
        rows.append({"setting": "Default RF with duplicates kept", "seed": seed, "cv_f1": f1,
                     "test_rows_also_in_train": leak, **test_evaluate({}, Xtr2, ytr2, Xte2, yte2, seed)})
    df = pd.DataFrame(rows)
    ref = final[final.method.isin(["Default", "TPE"])].assign(
        setting=lambda d: d.method.map({"Default": "Default RF (dedup, main)", "TPE": "TPE with penalty (main)"}))
    df = pd.concat([ref, df], ignore_index=True)
    df.to_csv(RESULTS / "ablation.csv", index=False)
    print(f"test rows that also appear in train when duplicates are kept: {leak:.1%}")
    print(df.groupby("setting", sort=False)[["cv_f1", "test_f1", "size_mb", "latency_ms_per_1k"]]
          .agg(["mean", "std"]).round(4).to_string())


def corrupt(X, rate, rng):
    """Replace a fraction `rate` of feature cells by a random value of the same feature (simulated noise)."""
    X = X.copy()
    mask = rng.random(X.shape) < rate
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        X[mask[:, j], j] = rng.choice(vals, mask[:, j].sum())
    return X


def robustness(Xtr, ytr, Xte, yte, final):
    rows = []
    for _, r in final[final.method.isin(["Default", "TPE"])].iterrows():
        params = {} if r.method == "Default" else best_params(r)
        m = RandomForestClassifier(**params, n_jobs=-1, random_state=int(r.seed)).fit(Xtr, ytr)
        for rate in (0.0, 0.05, 0.10, 0.20):
            Xn = corrupt(Xte, rate, np.random.default_rng(1000 + int(r.seed)))
            rows.append({"method": r.method, "seed": int(r.seed), "noise_rate": rate,
                         "test_f1": f1_score(yte, m.predict(Xn))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "robustness.csv", index=False)
    g = df.groupby(["method", "noise_rate"])["test_f1"].agg(["mean", "std"]).reset_index()
    print(g.round(4).to_string())
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in ("Default", "TPE"):
        d = g[g.method == m]
        ax.errorbar(d.noise_rate * 100, d["mean"], yerr=d["std"], color=plots.COLORS[m], marker="o", ms=8,
                    capsize=4, label=f"{m} (mean ± std, 10 seeds)")
    ax.set(xlabel="Corrupted feature cells in test set (%)", ylabel="Test F1",
           title="Robustness to noisy features")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "robustness_noise.png", dpi=150)


def plot_scalability(g):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, col, label in ((axes[0], "search_s", "TPE search time, 40 evaluations (s)"),
                           (axes[1], "mean_trial_n_estimators", "Mean n_estimators tried by TPE")):
        ax.errorbar(g.index, g[(col, "mean")], yerr=g[(col, "std")], color=plots.COLORS["TPE"], marker="o",
                    ms=8, capsize=4)
        ax.set(xlabel="Training rows n", title=label)
        ax.set_ylim(bottom=0)
    fig.text(0.5, 0.01, "mean ± std over 3 seeds, S_max = 1 MB", ha="center", fontsize=9, color=plots.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "scalability.png", dpi=150)


def scalability(Xtr, ytr, Xte, yte):
    rows = []
    for frac in (0.25, 0.5, 1.0):
        for seed in range(3):
            n = int(len(Xtr) * frac)
            idx = np.random.default_rng(seed).permutation(len(Xtr))[:n] if frac < 1 else np.arange(len(Xtr))
            t = time.perf_counter()
            best, trials = run_search("TPE", seed, TRIALS, Xtr[idx], ytr[idx])
            search_s = time.perf_counter() - t
            t = time.perf_counter()
            RandomForestClassifier(n_jobs=-1, random_state=seed).fit(Xtr[idx], ytr[idx])
            br = max(trials, key=lambda r: r["J"])
            rows.append({"train_fraction": frac, "n_train": n, "seed": seed, "search_s": search_s,
                         "mean_eval_s": np.mean([r["eval_s"] for r in trials]),
                         "mean_trial_n_estimators": np.mean([r["n_estimators"] for r in trials]),
                         "best_cv_f1": br["cv_f1"], "best_n_estimators": br["n_estimators"],
                         "best_size_mb": br["size_mb"],
                         "default_fit_s": time.perf_counter() - t})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "scalability.csv", index=False)
    g = df.groupby("n_train")[["search_s", "mean_eval_s", "mean_trial_n_estimators", "best_n_estimators",
                               "best_size_mb", "best_cv_f1", "default_fit_s"]].agg(["mean", "std"])
    print(g.round(4).to_string())
    plot_scalability(g)


def feature_importance(Xtr, ytr, final):
    names = pd.read_csv(DATA_CSV, nrows=0).columns[:-1]
    r = final[final.method == "TPE"].sort_values("J", ascending=False).iloc[0]  # chosen by CV J, not test
    m = RandomForestClassifier(**best_params(r), n_jobs=-1, random_state=int(r.seed)).fit(Xtr, ytr)
    df = pd.DataFrame({"feature": names, "importance": m.feature_importances_}).sort_values("importance", ascending=False)
    df.to_csv(RESULTS / "feature_importance.csv", index=False)
    print(df.head(10).round(4).to_string(index=False))
    top = df.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(top.feature, top.importance, color=plots.COLORS["TPE"], height=0.6)
    ax.set(xlabel="Mean decrease in impurity", title=f"Top-10 features of the chosen model ({model_size_mb(m):.2f} MB)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance.png", dpi=150)


def replot():
    """Redraw sensitivity/scalability figures from the saved CSVs (no recomputation)."""
    final = pd.read_csv(RESULTS / "final.csv")
    sens = pd.read_csv(RESULTS / "sensitivity.csv")
    plot_sensitivity(sens.groupby("s_max_mb")[["test_f1", "size_mb"]].agg(["mean", "std"]), final)
    sc = pd.read_csv(RESULTS / "scalability.csv")
    plot_scalability(sc.groupby("n_train")[["search_s", "mean_trial_n_estimators"]].agg(["mean", "std"]))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
    plt = plots.plt
    if sys.argv[1:] == ["--replot"]:
        replot()
        sys.exit()
    X, y, _ = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
    final = pd.read_csv(RESULTS / "final.csv")
    assert set(SPACE) == set(P)
    for step in (lambda: feature_importance(Xtr, ytr, final), lambda: robustness(Xtr, ytr, Xte, yte, final),
                 lambda: scalability(Xtr, ytr, Xte, yte), lambda: ablation(Xtr, ytr, Xte, yte, final),
                 lambda: sensitivity(Xtr, ytr, Xte, yte, final)):
        step()
