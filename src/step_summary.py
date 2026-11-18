"""Summarise the one-factor-at-a-time search V0 -> V10: what each hyperparameter change bought, and the final decision.

Reads results/ofat/versions.csv (V0-V4) and results/ofat_extra/versions.csv (V5-V10).
Writes results/step_summary.csv, results/decision.json and figures/ofat_steps.png.
"""
import json

import numpy as np
import pandas as pd

from experiment import RESULTS

PARAMS = ["n_estimators", "max_depth", "min_samples_leaf", "max_features",
          "criterion", "bootstrap", "min_samples_split", "max_leaf_nodes", "class_weight", "ccp_alpha"]
DEFAULTS = {"criterion": "gini", "bootstrap": True, "min_samples_split": 2, "max_leaf_nodes": None,
            "class_weight": None, "ccp_alpha": 0.0}  # V0-V4 rows do not carry these columns
METRICS = ["cv_f1", "test_f1", "test_precision", "test_recall", "size_mb", "latency_ms_per_1k"]


def norm(v):
    """One printable form per value (None/NaN, 11 vs 11.0, True vs 'True')."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "None"
    if isinstance(v, (bool, np.bool_)) or v in ("True", "False"):  # before float(): float(True) == 1.0
        return str(v)
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def to_kwargs(params):
    """Inverse of norm() for RF keyword arguments, e.g. {"max_depth": "None", "bootstrap": "False"}."""
    out = {}
    for k, v in params.items():
        v = norm(v)
        if v in ("None", "True", "False"):
            out[k] = {"None": None, "True": True, "False": False}[v]
        else:
            try:
                out[k] = float(v) if k in ("max_features", "ccp_alpha") else int(v)
            except ValueError:
                out[k] = v  # "sqrt", "gini", "balanced", ...
    return out


def load():
    a = pd.read_csv(RESULTS / "ofat" / "versions.csv")
    b = pd.read_csv(RESULTS / "ofat_extra" / "versions.csv")
    b = b[b.version != "V4"]  # V4 is the start point of the second run, already in a
    for k, v in DEFAULTS.items():
        a[k] = v
    df = pd.concat([a, b], ignore_index=True)
    df["vnum"] = df.version.str[1:].astype(int)
    return df.sort_values(["seed", "vnum"]).reset_index(drop=True)


def summarise(df):
    rows = []
    for vnum, g in df.groupby("vnum"):
        name = g.changed.iloc[0]
        row = {"version": f"V{vnum}", "changed": name}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std()
        if vnum > 0:
            prev = df[df.vnum == vnum - 1].set_index("seed")
            cur = g.set_index("seed")
            for m in METRICS:
                row[f"delta_{m}"] = (cur[m] - prev[m]).mean()  # mean of per-seed changes
            moved = cur[name].map(norm) != prev[name].map(norm)
            row["seeds_changed"] = int(moved.sum())
            row["values_chosen"] = "; ".join(f"{v}: {n}" for v, n in cur[name].map(norm).value_counts().items())
        rows.append(row)
    return pd.DataFrame(rows)


def decision(df):
    """Final configuration: the V10 run with the best CV objective (the test set plays no part in the choice)."""
    v10 = df[df.vnum == df.vnum.max()]
    best = v10.sort_values("J", ascending=False).iloc[0]
    return {"rule": "V10 run with the highest CV F1 over 10 seeds (test set not used for the choice)",
            "seed": int(best.seed), "params": {k: norm(best[k]) for k in PARAMS},
            "chosen_model": {m: round(float(best[m]), 4) for m in METRICS},
            "v10_mean_over_seeds": {m: round(float(v10[m].mean()), 4) for m in METRICS},
            "v10_std_over_seeds": {m: round(float(v10[m].std()), 4) for m in METRICS}}


def compare_methods(df):
    """Same objective, same 4 hyperparameters, same 40-evaluation budget: Default / Random / TPE vs OFAT V4;
    plus OFAT V10 (6 more hyperparameters, +24 evaluations). Paired Wilcoxon over the 10 seeds."""
    from scipy.stats import wilcoxon
    final = pd.read_csv(RESULTS / "final.csv")
    ext = pd.read_csv(RESULTS / "ofat_extra" / "versions.csv").groupby("seed").search_s.first()
    v4 = df[df.vnum == 4].set_index("seed")
    v10 = df[df.vnum == 10].set_index("seed")
    runs = {m: final[final.method == m].set_index("seed") for m in ("Default", "Random", "TPE")}
    runs["OFAT V4"] = v4
    runs["OFAT V10"] = v10.assign(search_s=v4.search_s + ext)  # total search time of both stages
    table = {m: {k: [round(float(d[k].mean()), 4), round(float(d[k].std()), 4)]
                 for k in ("cv_f1", "test_f1", "test_precision", "test_recall", "size_mb", "latency_ms_per_1k", "search_s")
                 if k in d and d[k].notna().any()} for m, d in runs.items()}
    pairs = [("OFAT V4", "TPE"), ("OFAT V4", "Random"), ("TPE", "Random"), ("OFAT V10", "Default"), ("TPE", "Default")]
    tests = {f"{a} vs {b}": {k: {"p": round(float(wilcoxon(runs[a][k], runs[b][k]).pvalue), 4),
                                 "wins": int((runs[a][k] > runs[b][k]).sum())} for k in ("cv_f1", "test_f1")}
             for a, b in pairs}
    # evaluations needed to beat the default RF's mean CV F1 (same bar as plots.stats)
    bar = round(float(runs["Default"].cv_f1.mean()), 4)
    ev = pd.read_csv(RESULTS / "ofat" / "evaluations.csv").sort_values(["seed", "eval"])
    hit = ev[ev.groupby("seed").J.cummax() > bar].groupby("seed")["eval"].min()
    out = {"metrics_mean_std": table, "wilcoxon": tests,
           "ofat_evals_to_beat_default_cv": {"bar": bar, "mean": float(hit.mean()), "reached_seeds": int(hit.size)}}
    (RESULTS / "method_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def plot(s):
    import plots  # shared style
    plt = plots.plt
    x = range(len(s))
    labels = [f"{v}\n{'default' if c == '-' else '+' + c}" for v, c in zip(s.version, s.changed)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col, color, label in (("cv_f1", plots.COLORS["Default"], "CV F1 (objective)"),
                              ("test_f1", plots.COLORS["TPE"], "Test F1")):
        axes[0].errorbar(x, s[f"{col}_mean"], yerr=s[f"{col}_std"], color=color, marker="o", ms=6, capsize=3, label=label)
    axes[0].set_title("F1 after each step (mean ± std, 10 seeds)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[1].errorbar(x, s.size_mb_mean, yerr=s.size_mb_std, color=plots.COLORS["Random"], marker="o", ms=6, capsize=3)
    axes[1].set_title("Model size (MB)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(list(x), labels, fontsize=7, rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS.parent / "figures" / "ofat_steps.png", dpi=150)


def main():
    df = load()
    s = summarise(df)
    s.to_csv(RESULTS / "step_summary.csv", index=False)
    d = decision(df)
    (RESULTS / "decision.json").write_text(json.dumps(d, indent=2))
    show = s[["version", "changed", "seeds_changed", "values_chosen", "cv_f1_mean", "delta_cv_f1", "test_f1_mean",
              "delta_test_f1", "delta_test_recall", "size_mb_mean", "delta_size_mb", "delta_latency_ms_per_1k"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(show.round(4).to_string(index=False))
    print(json.dumps(d, indent=2))
    compare_methods(df)
    plot(s)


if __name__ == "__main__":
    main()
