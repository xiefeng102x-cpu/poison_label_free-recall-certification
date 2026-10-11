"""Certified recall vs declared poison-count ceiling, from the independent check draws
(score-table analysis only).  recall_lo(rho) = caught_lo / ceil(rho * N_p); caught_lo
does not depend on rho because the BetaBinomial bound uses n - N_min only.
Usage: python analysis/plot_certified_recall.py [delta]   (default 0.025)"""
import sys, pathlib, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
delta = sys.argv[1] if len(sys.argv) > 1 else "0.025"
D_DIR = ROOT / "results" / "bloodmnist" / f"independent_check_delta{delta}"
MANUSCRIPT_FIG = ROOT / "figs" / "fig_certified_recall_vs_ceiling"
D = pd.read_csv(D_DIR / "draws.csv"); C = pd.read_csv(D_DIR / "counts.csv")
D = D.merge(C[["view", "seed", "fold", "poisons"]], on=["view", "seed", "fold"])
TAU = 0.8
rhos = np.round(np.arange(1.0, 2.51, 0.025), 3)
rows = []
for (view, k), g in D.groupby(["view", "k"]):
    for rho in rhos:
        rl = np.minimum(g.caught_lo / np.ceil(rho * g.poisons), 1.0)
        rows.append(dict(view=view, k=k, rho=rho, certified_recall_mean=rl.mean(),
                         certified_recall_p05=rl.quantile(0.05), true_recall=g.recall.mean(),
                         go_rate=(rl >= TAU).mean()))
P = pd.DataFrame(rows); P.to_csv(D_DIR / "certified_recall_curve.csv", index=False)

C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, k in zip(axes, (120, 500)):
    for view, col, lab in (("paired_hidden64_drift", C1, "paired drift"), ("suspect_hidden64", C2, "suspect hidden")):
        s = P[(P.view == view) & (P.k == k)]
        ax.plot(s.rho, s.certified_recall_mean, color=col, lw=2, label=f"{lab}: certified (mean)")
        ax.fill_between(s.rho, s.certified_recall_p05, s.certified_recall_mean, color=col, alpha=0.12, lw=0)
        ax.axhline(s.true_recall.iloc[0], color=col, lw=1, ls=(0, (1, 1.5)))
    ax.axhline(TAU, color=INK, lw=1)
    ax.text(2.48, TAU + 0.01, "τ = 0.8", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.axvline(1.25, color=INK2, lw=0.8, ls=(0, (4, 2)))
    ax.text(1.27, 0.32, "cap for a perfect\ndetector: 1/τ = 1.25", fontsize=7, color=INK2)
    panel = "(a)" if k == 120 else "(b)"
    ax.set_title(f"{panel} k = {k} trusted members per fold", fontsize=8.5, color=INK2, loc="left")
    ax.set_xlabel("declared poison-count ceiling ÷ true count")
axes[0].set_ylabel("certified recall lower bound")
axes[0].set_ylim(0.3, 1.02)
axes[1].legend(fontsize=7, loc="upper right")
fig.suptitle("BloodMNIST BadNets, 5 archived victims × 2 folds (retrospective); dotted = true recall, band = 5th pct",
             fontsize=8, color=INK2, x=0.02, y=1.04, ha="left")
for output in (D_DIR / "fig_certified_recall_vs_ceiling", MANUSCRIPT_FIG):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png")); fig.savefig(output.with_suffix(".pdf"))
print(P[P.rho.isin([1.0, 1.1, 1.2, 1.5, 2.0])].round(3).to_string(index=False))
