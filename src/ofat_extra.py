"""Continue the one-factor-at-a-time search from V4 with RF hyperparameters left at default so far (V5-V10).

V5 criterion -> V6 bootstrap -> V7 min_samples_split -> V8 max_leaf_nodes -> V9 class_weight -> V10 ccp_alpha
Each is changed at most once (kept if no grid value beats the current one), same objective J and 1 MB budget.
Run src/ofat_search.py first. Output: results/ofat_extra/ and figures/ofat_extra_versions.png.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment import LAMBDA, RESULTS, S_MAX, SPLIT_SEED, TEST_SIZE, cv_evaluate, load_data, penalized, test_evaluate
from ofat_search import ORDER as BASE_ORDER, coordinate_search, figdir

EXTRA = {  # first value = sklearn default (= the current value at V4)
    "criterion": ["gini", "entropy"],  # log_loss is the same split criterion as entropy in sklearn
    "bootstrap": [True, False],
    "min_samples_split": [2, 4, 8, 16, 32],
    "max_leaf_nodes": [None, 16, 32, 64, 128, 256],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "ccp_alpha": [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
}


def v4_params(row):
    """Parse a V4 row of results/ofat/versions.csv back into RF keyword arguments."""
    mf = row.max_features
    mf = mf if mf in ("sqrt", "log2") else float(mf)
    depth = None if pd.isna(row.max_depth) else int(float(row.max_depth))
    return {"n_estimators": int(row.n_estimators), "max_depth": depth,
            "min_samples_leaf": int(row.min_samples_leaf), "max_features": mf}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(RESULTS / "ofat"), help="folder of the V0-V4 run to continue from")
    ap.add_argument("--out", default=str(RESULTS / "ofat_extra"))
    ap.add_argument("--lam", type=float, default=LAMBDA, help="size-penalty weight; 0 = no size budget")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
    out = pd.io.common.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v4 = pd.read_csv(pd.io.common.Path(a.src) / "versions.csv").query("version == 'V4'").set_index("seed")
    X, y, _ = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
    order = list(EXTRA)

    versions_all, logs_all = [], []
    for seed in v4.index:
        start = {**v4_params(v4.loc[seed]), **{k: g[0] for k, g in EXTRA.items()}}

        def evaluate(p):
            f1, size = cv_evaluate(p, Xtr, ytr, int(seed))
            return penalized(f1, size, S_MAX, a.lam), {"cv_f1": f1, "size_mb": size}
        t = time.perf_counter()
        versions, log = coordinate_search(start, EXTRA, order, evaluate)
        search_s = time.perf_counter() - t
        for i, v in enumerate(versions):
            v["version"] = f"V{4 + i}"
            p = {k: v[k] for k in BASE_ORDER + order}
            res = test_evaluate(p, Xtr, ytr, Xte, yte, int(seed))
            versions_all.append({"seed": int(seed), **v, "search_s": search_s, "feasible": v["size_mb"] <= S_MAX,
                                 **{k: res[k] for k in ("test_f1", "test_precision", "test_recall", "latency_ms_per_1k")}})
        logs_all += [{"seed": int(seed), "eval": i + 1, **r} for i, r in enumerate(log)]
        last = versions[-1]
        print(f"seed {seed}: V10 extra = {[last[k] for k in order]} J={last['J']:.4f} size={last['size_mb']:.2f}MB "
              f"({search_s:.0f}s)")

    vdf, ldf = pd.DataFrame(versions_all), pd.DataFrame(logs_all)
    vdf.to_csv(out / "versions.csv", index=False)
    ldf.to_csv(out / "evaluations.csv", index=False)
    cols = ["J", "cv_f1", "test_f1", "test_recall", "size_mb", "latency_ms_per_1k"]
    summary = vdf.groupby(["version", "changed"], sort=False)[cols].agg(["mean", "std"])
    summary["feasible_rate"] = vdf.groupby(["version", "changed"], sort=False)["feasible"].mean()
    summary.to_csv(out / "summary.csv")
    print(summary.round(4).to_string())
    report_changes(vdf)
    plot(vdf, out)


def report_changes(vdf):
    last = vdf[vdf.version == "V10"]
    print(f"\nhow often each extra hyperparameter moved away from its default (out of {len(last)} seeds):")
    for k, g in EXTRA.items():
        vals = last[k].map(lambda v: "None" if pd.isna(v) else str(v))  # None is written to CSV as an empty cell
        print(f"  {k:18s} changed in {(vals != str(g[0])).sum():2d}/{len(last)}   values: {vals.value_counts().to_dict()}")


def plot(vdf, out):
    import plots  # shared style; imported late because it reads sys.argv
    plt = plots.plt
    short = {"-": "start", "criterion": "+criterion", "bootstrap": "+bootstrap", "min_samples_split": "+min_split",
             "max_leaf_nodes": "+max_leaves", "class_weight": "+class_w", "ccp_alpha": "+ccp_alpha"}
    g = vdf.groupby("version", sort=False)[["J", "test_f1", "size_mb"]].agg(["mean", "std"])
    labels = [f"{v}\n{short[c]}" for v, c in vdf.groupby("version", sort=False).changed.first().items()]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, col, title in ((axes[0], "J", "CV objective J"), (axes[1], "test_f1", "Test F1"),
                           (axes[2], "size_mb", "Model size (MB)")):
        ax.errorbar(range(len(g)), g[(col, "mean")], yerr=g[(col, "std")], color=plots.COLORS["Default"],
                    marker="o", ms=6, capsize=4)
        ax.set_xticks(range(len(g)), labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(title)
    axes[2].axhline(S_MAX, color=plots.MUTED, ls=":", lw=1.2)
    axes[2].set_ylim(bottom=0)
    fig.text(0.5, 0.01, "mean ± std over 10 seeds; each step changes one more hyperparameter (kept if nothing better)",
             ha="center", fontsize=8, color=plots.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figdir(out) / "ofat_extra_versions.png", dpi=150)
    print("figure written to", figdir(out))


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--replot"]:  # re-print the change counts and redraw from saved CSVs
        o = pd.io.common.Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "ofat_extra"
        sys.argv = sys.argv[:1]
        v = pd.read_csv(o / "versions.csv")
        report_changes(v)
        plot(v, o)
    else:
        main()
