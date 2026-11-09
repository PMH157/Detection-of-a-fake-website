"""Build figures from results/*.csv (no recomputation)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# optional arg: a results dir from a demo run; its figures go to <dir>/figures
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
FIG = RES / "figures" if len(sys.argv) > 1 else ROOT / "figures"
COLORS = {"Default": "#2a78d6", "Random": "#eb6834", "TPE": "#1baf7a"}  # fixed categorical order
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 10, "lines.linewidth": 2,
})


def convergence(trials, s_max):
    fig, ax = plt.subplots(figsize=(7, 4))
    lows, tops = [], []
    for m in ("Random", "TPE"):
        d = trials[trials.method == m].sort_values(["seed", "trial"])
        best = d.groupby("seed")["J"].cummax()
        curve = best.groupby(d["trial"]).agg(["mean", "std"]).fillna(0)
        curve = curve.iloc[min(2, len(curve) - 1):]  # first 2 trials: J << 0, huge std
        ax.plot(curve.index + 1, curve["mean"], color=COLORS[m], label=f"{m} (mean ± std, {d.seed.nunique()} seeds)")
        ax.fill_between(curve.index + 1, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=COLORS[m], alpha=0.15, lw=0)
        lows.append((curve["mean"] - curve["std"]).min())
        tops.append(curve["mean"].max())
    d0 = trials[trials.method == "Default"]
    top = max(tops)
    ax.axvline(10.5, color=MUTED, ls=":", lw=1.2)
    ax.text(10.7, top + 0.005, "TPE starts modelling\n(trials 1–10 = same random draws)", color=MUTED, fontsize=8, va="top")
    ax.set_ylim(min(lows) - 0.006, top + 0.011)
    ax.text(0.98, 0.97, ha="right", va="top", s=f"Default RF: CV F1 = {d0.cv_f1.mean():.4f}, size = {d0.size_mb.mean():.1f} MB\n"
            f"→ violates S_max, J = {d0.J.mean():.2f} (off-scale)", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    ax.set(xlabel="Evaluation (trial)", ylabel="Best J(x) so far  (CV F1 − size penalty)",
           title=f"Convergence of the search  (size budget {s_max} MB)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png", dpi=150)


def pareto_front(d):
    d = d.sort_values(["size_mb", "cv_f1"], ascending=[True, False])
    return d[d["cv_f1"] > d["cv_f1"].cummax().shift(fill_value=-1)]


def tradeoff(trials, s_max):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in ("Random", "TPE", "Default"):
        d = trials[trials.method == m]
        ax.scatter(d.size_mb, d.cv_f1, s=40 if m == "Default" else 14, color=COLORS[m],
                   alpha=1 if m == "Default" else 0.45, edgecolors="#fcfcfb", linewidths=0.6,
                   marker="D" if m == "Default" else "o", label=m, zorder=3 if m == "Default" else 2)
    pf = pareto_front(trials[trials.method != "Default"])
    ax.step(pf.size_mb, pf.cv_f1, where="post", color=INK, lw=1.2, label="Pareto front (all searched x)", zorder=4)
    ax.axvline(s_max, color=MUTED, ls=":", lw=1.5)
    ax.set_xscale("log")
    ax.set(xlabel="Model size (MB, log scale)", ylabel="CV F1 (train folds only)",
           title="Trade-off: detection quality vs model size")
    ax.set_ylim(bottom=max(ax.get_ylim()[0], trials.cv_f1.quantile(0.05) - 0.01))
    ax.text(s_max * 1.08, ax.get_ylim()[0] + 0.002, f"S_max = {s_max} MB\n(infeasible →)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff_pareto.png", dpi=150)


def test_box(final):
    methods = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, col, label in ((axes[0], "test_f1", "Test F1"), (axes[1], "size_mb", "Model size (MB)")):
        data = [final[final.method == m][col] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK})
        ax.set_xticks(range(1, len(methods) + 1), methods)
        for patch, m in zip(bp["boxes"], methods):
            patch.set(facecolor=COLORS[m], alpha=0.6, edgecolor=MUTED)
        ax.set_title(f"{label} across seeds")
    fig.tight_layout()
    fig.savefig(FIG / "test_boxplot.png", dpi=150)


def stats(trials, final, threshold=0.94):
    """Paired Wilcoxon (same seed) TPE vs Random, and evaluations needed to reach J >= threshold."""
    from scipy.stats import wilcoxon
    piv = final.pivot(index="seed", columns="method")
    out = ({f"wilcoxon_p_{c}": float(wilcoxon(piv[c]["TPE"], piv[c]["Random"]).pvalue) for c in ("J", "test_f1")}
           if len(piv) >= 2 else {})  # a paired test needs at least 2 seeds
    for m in ("Random", "TPE"):
        d = trials[trials.method == m].sort_values(["seed", "trial"])
        hit = d[d.groupby("seed")["J"].cummax() >= threshold].groupby("seed")["trial"].min() + 1
        out[f"{m}_evals_to_J>={threshold}"] = {"mean": float(hit.mean()), "reached_seeds": int(hit.size),
                                               "of_seeds": int(d.seed.nunique())}
    (RES / "stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    FIG.mkdir(exist_ok=True)
    s_max = json.loads((RES / "config.json").read_text())["s_max_mb"]
    trials, final = pd.read_csv(RES / "trials.csv"), pd.read_csv(RES / "final.csv")
    convergence(trials, s_max)
    tradeoff(trials, s_max)
    test_box(final)
    stats(trials, final)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
