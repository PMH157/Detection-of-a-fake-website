"""One-factor-at-a-time (coordinate) search, starting from the default RF (steps V0-V4 of the final decision).

Each hyperparameter is changed exactly once, in a fixed order; each step is its own version:
  V0 default -> V1 tune n_estimators -> V2 + max_depth -> V3 + min_samples_leaf -> V4 + max_features
At each step 10 grid values are tried with the others frozen (same objective J and 1 MB budget as
the main experiment); the current value is kept if nothing beats it. 4 x 10 = 40 evaluations, the
same budget as Random Search / TPE.

Output: results/ofat/ (CSV) and figures/ofat_convergence.png, figures/ofat_versions.png.
Continue with src/ofat_extra.py (steps V5-V10).
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment import (LAMBDA, RESULTS, S_MAX, SPACE, SPLIT_SEED, TEST_SIZE, cv_evaluate, load_data, penalized,
                        test_evaluate)

ORDER = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]
DEFAULT = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}  # sklearn defaults


def grids(k=10):
    lo, hi = SPACE["n_estimators"]
    return {
        "n_estimators": sorted({int(round(v)) for v in np.geomspace(lo, hi, k)}),  # log scale like the main search
        "max_depth": [int(round(v)) for v in np.linspace(*SPACE["max_depth"], k)],
        "min_samples_leaf": [int(round(v)) for v in np.linspace(*SPACE["min_samples_leaf"], k)],
        "max_features": [round(float(v), 3) for v in np.linspace(*SPACE["max_features"], k)],
    }


def coordinate_search(start, grid, order, evaluate):
    """One pass of coordinate search. evaluate(params) -> (J, info). Returns (versions, log)."""
    cur = dict(start)
    cur_j, info = evaluate(cur)
    versions = [{"version": "V0", "changed": "-", **cur, "J": cur_j, **info}]
    log = []
    for step, name in enumerate(order, 1):
        best_v, best_j, best_info = cur[name], cur_j, info  # keep current value unless a grid value beats it
        for v in grid[name]:
            cand = {**cur, name: v}
            j, inf = evaluate(cand)
            log.append({"step": step, "param": name, "value": v, "J": j, **inf})
            if j > best_j:
                best_v, best_j, best_info = v, j, inf
        cur[name], cur_j, info = best_v, best_j, best_info
        versions.append({"version": f"V{step}", "changed": name, **cur, "J": cur_j, **info})
    return versions, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS / "ofat"))
    ap.add_argument("--lam", type=float, default=LAMBDA, help="size-penalty weight; 0 = no size budget")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
    out = pd.io.common.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
    G = grids()
    print("grids:", G)

    all_versions, all_logs = [], []
    for seed in range(a.seeds):
        def evaluate(p):
            f1, size = cv_evaluate(p, Xtr, ytr, seed)
            return penalized(f1, size, S_MAX, a.lam), {"cv_f1": f1, "size_mb": size}
        t = time.perf_counter()
        versions, log = coordinate_search(DEFAULT, G, ORDER, evaluate)
        search_s = time.perf_counter() - t
        for v in versions:  # run every version on its own: refit on train, evaluate once on test
            p = {k: v[k] for k in ORDER}
            res = test_evaluate(p, Xtr, ytr, Xte, yte, seed)
            all_versions.append({"seed": seed, **v, "search_s": search_s, **{k: res[k] for k in
                                 ("test_f1", "test_precision", "test_recall", "latency_ms_per_1k")},
                                 "feasible": v["size_mb"] <= S_MAX})
        all_logs += [{"seed": seed, "eval": i + 1, **r} for i, r in enumerate(log)]
        last = versions[-1]
        print(f"seed {seed}: V4 = {[last[k] for k in ORDER]}  J={last['J']:.4f} size={last['size_mb']:.2f}MB ({search_s:.0f}s)")

    vdf, ldf = pd.DataFrame(all_versions), pd.DataFrame(all_logs)
    vdf.to_csv(out / "versions.csv", index=False)
    ldf.to_csv(out / "evaluations.csv", index=False)
    cols = ["J", "cv_f1", "test_f1", "test_recall", "size_mb", "latency_ms_per_1k"]
    summary = vdf.groupby(["version", "changed"], sort=False)[cols].agg(["mean", "std"])
    summary["feasible_rate"] = vdf.groupby(["version", "changed"], sort=False)["feasible"].mean()
    summary.to_csv(out / "summary.csv")
    print(summary.round(4).to_string())
    print("\nchosen values per seed (V4):")
    print(vdf[vdf.version == "V4"][["seed"] + ORDER].to_string(index=False))
    (out / "meta.json").write_text(f'{{"lambda": {a.lam}, "s_max_mb": {S_MAX}}}')
    plot(vdf, ldf, out, compare=a.lam == LAMBDA)


def figdir(out):
    """Main run (results/...) -> figures/; any other output folder keeps its figures next to its CSVs."""
    f = RESULTS.parent / "figures" if out.resolve().parent == RESULTS.resolve() else out
    f.mkdir(exist_ok=True)
    return f


def plot(vdf, ldf, out, compare=True):
    import plots  # shared style; imported late because it reads sys.argv
    plt = plots.plt
    c = plots.COLORS

    # Convergence: best-so-far J vs evaluation, OFAT next to Random/TPE from the main experiment
    fig, ax = plt.subplots(figsize=(7, 4))
    base = vdf[vdf.version == "V0"].set_index("seed").J
    curves = {"OFAT (coordinate)": ldf.assign(J=ldf.J).sort_values(["seed", "eval"])}
    lows, tops = [], []
    trials = pd.read_csv(RESULTS / "trials.csv")
    series = [("Random", c["Random"], trials[trials.method == "Random"], "trial"),
              ("TPE", c["TPE"], trials[trials.method == "TPE"], "trial"),
              ("OFAT (coordinate)", c["Default"], curves["OFAT (coordinate)"], "eval")]
    # Random/TPE logs were optimised under the 1 MB budget; only overlay them when OFAT used the same objective
    for name, color, d, xcol in (series if compare else series[2:]):
        d = d.sort_values(["seed", xcol])
        best = d.groupby("seed")["J"].cummax()
        if name.startswith("OFAT"):  # OFAT also has the V0 point evaluated before its 40 grid evaluations
            best = np.maximum(best, d.seed.map(base))
        cur = best.groupby(d[xcol].values).agg(["mean", "std"]).fillna(0)
        x = cur.index + (0 if name.startswith("OFAT") else 1)
        cur, x = cur.iloc[2:], x[2:]
        ax.plot(x, cur["mean"], color=color, label=f"{name} (mean ± std, 10 seeds)")
        ax.fill_between(x, cur["mean"] - cur["std"], cur["mean"] + cur["std"], color=color, alpha=0.15, lw=0)
        lows.append((cur["mean"] - cur["std"]).min())
        tops.append(cur["mean"].max())
    for k in (10.5, 20.5, 30.5):
        ax.axvline(k, color=plots.MUTED, ls=":", lw=1)
    for i, name in enumerate(["n_estimators", "max_depth", "min_samples_leaf", "max_features"]):
        ax.text(5.5 + 10 * i, max(tops) + 0.004, f"OFAT step {i + 1}:\n{name}", ha="center", va="center",
                fontsize=7.5, color=plots.MUTED)
    ax.set_ylim(min(lows) - 0.003, max(tops) + 0.008)
    title = ("One-factor-at-a-time vs Random vs TPE (40 evaluations)" if compare
             else "One-factor-at-a-time, no size budget (objective = CV F1)")
    ax.set(xlabel="Evaluation", ylabel="Best J(x) so far", title=title)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir(out) / "ofat_convergence.png", dpi=150)

    # Per-version test F1 and size (two panels, no dual axis)
    g = vdf.groupby("version", sort=False)[["test_f1", "size_mb"]].agg(["mean", "std"])
    short = {"-": "default", "n_estimators": "+trees", "max_depth": "+depth", "min_samples_leaf": "+leaf",
             "max_features": "+max_feat"}
    labels = [f"{v}\n{short[ch]}" for v, ch in vdf.groupby("version", sort=False).changed.first().items()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, title in ((axes[0], "test_f1", "Test F1"), (axes[1], "size_mb", "Model size (MB, log scale)")):
        ax.errorbar(range(len(g)), g[(col, "mean")], yerr=g[(col, "std")], color=c["Default"], marker="o", ms=7, capsize=4)
        ax.set_xticks(range(len(g)), labels, fontsize=8)
        ax.set_title(title)
    axes[1].axhline(S_MAX, color=plots.MUTED, ls=":", lw=1.2)
    axes[1].text(len(g) - 1, S_MAX * 1.1, f"S_max = {S_MAX} MB", ha="right", fontsize=8, color=plots.MUTED)
    axes[1].set_yscale("log")
    lo = min((g[("size_mb", "mean")] - g[("size_mb", "std")]).clip(lower=0.1).min(), S_MAX)
    hi = (g[("size_mb", "mean")] + g[("size_mb", "std")]).max()
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64) if lo / 2 <= t <= hi * 2]  # only ticks near the data
    axes[1].set_yticks(ticks, [f"{t:g}" for t in ticks])
    axes[1].minorticks_off()
    fig.tight_layout()
    fig.savefig(figdir(out) / "ofat_versions.png", dpi=150)
    print("figures written to", figdir(out))


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--replot"]:  # redraw from saved CSVs without rerunning the search
        import json
        o = pd.io.common.Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "ofat"
        sys.argv = sys.argv[:1]
        meta = json.loads((o / "meta.json").read_text()) if (o / "meta.json").exists() else {"lambda": LAMBDA}
        plot(pd.read_csv(o / "versions.csv"), pd.read_csv(o / "evaluations.csv"), o, compare=meta["lambda"] == LAMBDA)
    else:
        main()
