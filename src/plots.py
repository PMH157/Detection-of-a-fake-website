"""Build figures from results/*.csv (no recomputation)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
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
    for m in ("Random", "TPE"):
        d = trials[trials.method == m].sort_values(["seed", "trial"])
        best = d.groupby("seed")["J"].cummax()
        curve = best.groupby(d["trial"]).agg(["mean", "std"])
        ax.plot(curve.index + 1, curve["mean"], color=COLORS[m], label=f"{m} (mean ± std, {d.seed.nunique()} seeds)")
        ax.fill_between(curve.index + 1, curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=COLORS[m], alpha=0.15, lw=0)
    d0 = trials[trials.method == "Default"]
    top = max(line.get_ydata().max() for line in ax.get_lines())
    # Early infeasible trials have J << 0; zoom on the region where the methods actually differ.
    ax.set_ylim(top - 0.04, top + 0.008)
    ax.text(0.02, 0.04, f"Default RF: CV F1 = {d0.cv_f1.mean():.4f}, size = {d0.size_mb.mean():.1f} MB\n"
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
    ax.text(s_max, ax.get_ylim()[0], f"  S_max = {s_max} MB", color=MUTED, va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.set(xlabel="Model size (MB, log scale)", ylabel="CV F1 (train folds only)",
           title="Trade-off: detection quality vs model size")
    ax.set_ylim(bottom=max(ax.get_ylim()[0], trials.cv_f1.quantile(0.05) - 0.01))
    ax.legend(frameon=False, loc="lower right")
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


def main():
    FIG.mkdir(exist_ok=True)
    import json
    s_max = json.load(open(RES / "config.json"))["s_max_mb"]
    trials, final = pd.read_csv(RES / "trials.csv"), pd.read_csv(RES / "final.csv")
    convergence(trials, s_max)
    tradeoff(trials, s_max)
    test_box(final)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
