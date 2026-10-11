"""Figures for the S0 report (synthetic results only)."""
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = pathlib.Path(os.environ.get("P14CERT_OUT", ROOT / "results"))
FIG = pathlib.Path(os.environ.get("P14CERT_FIG", ROOT / "figs"))
FIG.mkdir(exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)

# Fig 1: false-assurance rate, certificate vs plug-in, and misspecified ceiling
d = pd.read_csv(RES / "s0a_main.csv")
d8 = d[d.tau == 0.8]
fig, ax = plt.subplots(figsize=(6.2, 2.8))
groups = [("Plug-in\n(oracle π, no CI)", d8[d8.range == "declared_exact"].plugin_far_pop, C[1]),
          ("Certificate\nπ exact", d8[d8.range == "declared_exact"].far_pop, C[0]),
          ("Certificate\nπ ∈ [π/2, 2π]", d8[d8.range == "declared_correct"].far_pop, C[0]),
          ("Certificate\nπ_max set too low", d8[d8.range == "declared_misspecified"].far_pop, C[3])]
for i, (lab, v, col) in enumerate(groups):
    v = np.asarray(v)
    x = i + (np.random.default_rng(i).random(v.size) - 0.5) * 0.5
    ax.scatter(x, v, s=6, color=col, alpha=0.35, linewidths=0)
    ax.plot([i - 0.3, i + 0.3], [v.max()] * 2, color=INK, lw=1.2)
    ax.text(i + 0.32, v.max() + (0.06 if v.max() < 0.1 else 0), f"max {v.max():.3f}", va="center", fontsize=7.5, color=INK2)
ax.axhline(0.05, color=INK2, lw=1, ls="-")
ax.text(3.45, 0.05, "δ = 0.05", va="bottom", ha="right", fontsize=7.5, color=INK2)
ax.grid(False); ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("false-assurance rate per cell"); ax.set_ylim(-0.02, 1.02)
ax.set_title("Each dot = one of 1,200 synthetic cells (4 ROC shapes × π × separation × k × n)",
             fontsize=8.5, color=INK2, loc="left")
save(fig, "fig1_false_assurance")

# Fig 2: transport amplification
t = pd.read_csv(RES / "s0a_transport.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
modes = [("fresh_eta0", "fresh calibration, no tolerance", C[1], "o"),
         ("trusted_members", "calibrate on trusted members", C[0], "s"),
         ("fresh_eta_point", "fresh + declared gap tolerance", C[2], "^")]
for ax, pi in zip(axes, [0.01, 0.05]):
    x = t[(t.mu == 1.5) & (t.k == 2000) & (t.pi == pi)]
    for m, lab, col, mk in modes:
        y = x[x["mode"] == m].sort_values("shift")
        gap = (y.mean_true_fpr - 0.05) * 100
        ax.plot(y["shift"], y["far_tau0.5"], color=col, marker=mk, ms=5, label=lab)
    ax.axhline(0.05, color=INK2, lw=1)
    ax.set_title(f"π = {pi:.0%}   (true TPR ≈ 0.43, target 0.5)", fontsize=8.5, color=INK2, loc="left")
    ax.set_xlabel("clean-candidate score shift s (σ units)\n≈ FPR gap of 0 / 0.5 / 1.1 / 2.4 points")
axes[0].set_ylabel("false-GO rate"); axes[0].legend(fontsize=7.5, loc="upper left")
save(fig, "fig2_transport")

# Fig 3: price of assurance
k = pd.read_csv(RES / "s0b_kstar.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for i, (mg, mk) in enumerate([(0.05, "o"), (0.10, "s"), (0.15, "^")]):
    s = k[k.margin == mg].sort_values("pi")
    ax.plot(s.pi, s.k_star_empirical, color=C[i], marker=mk, ms=5, label=f"margin {mg:.2f}: certificate (simulated)")
s = k[(k.margin == 0.05) & (k.tail_condition_holds)].sort_values("pi")
ax.plot(s.pi, s.k_lower_bound_test, color=INK, lw=1.2, ls=(0, (4, 2)),
        label="margin 0.05: lower bound for ANY GO rule\n(blue curve is ≈4× above it)")
ax.axhspan(80, 120, color=C[3], alpha=0.15, lw=0)
ax.text(0.011, 135, "paper14 trusted calibration: 80–120", fontsize=7.5, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("poison prevalence π"); ax.set_ylabel("trusted clean samples k needed\nto certify TPR ≥ 0.8 (power 0.8)")
ax.legend(fontsize=6.8, loc="upper right")
save(fig, "fig3_price_of_assurance")

# Fig 4: prevalence-ceiling tolerance
c = pd.read_csv(RES / "s0b_ceiling.csv")
fig, ax = plt.subplots(figsize=(4.6, 2.8))
for i, (tp, mk) in enumerate([(0.85, "o"), (0.95, "s"), (0.99, "^")]):
    s = c[c.tpr_true == tp].sort_values("ceiling_ratio")
    ax.plot(s.ceiling_ratio, s.p_go, color=C[i], marker=mk, ms=4, label=f"true TPR {tp}")
    pr = s.predicted_max_ratio.iloc[0]
    ax.axvline(pr, color=C[i], lw=1, ls=(0, (1, 1.5)))
ax.set_xlabel("declared prevalence ceiling ÷ true prevalence (π_max / π)")
ax.set_ylabel("P(GO), k = 10⁷, n = 10⁸")
ax.set_title("dotted: predicted limit (TPR − α)/(τ − α)", fontsize=8.5, color=INK2, loc="left")
ax.legend(fontsize=7.5)
save(fig, "fig4_ceiling_tolerance")

# Fig 5: direction budget collapse
e = pd.read_csv(RES / "s0c_direction_budget.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
for ax, spec in zip(axes, ["isotropic", "anisotropic"]):
    x = e[(e.spectrum == spec) & (e.pi_r == 1.0)]
    for i, dd in enumerate([8, 16, 32, 64]):
        if i > 3: break
        s = x[(x.d == dd) & (x.pi == 0.2)].sort_values("m")
        col = C[i]
        ax.plot(s.m_over_mstar, s.tpr_sim, color=col, marker="o", ms=3.5, label=f"d = {dd}")
        ax.plot(s.m_over_mstar, s.tpr_pred, color=col, lw=1, ls=(0, (1, 1.5)))
    ax.set_xscale("log"); ax.set_xlabel("reference size m ÷ m*,  m* = tr(Σ)/(π‖Δ‖)²")
    ax.set_title(f"{spec} Σ", fontsize=8.5, color=INK2, loc="left")
axes[0].set_ylabel("TPR at FPR 5%"); axes[0].legend(fontsize=7.5)
fig.suptitle("π = 0.2, π·r = 1; solid = simulation, dotted = prediction", fontsize=8.5, color=INK2, x=0.02, y=1.05, ha="left")
save(fig, "fig5_direction_budget")
print("figures written to", FIG)

# Fig 6: realized-recall certificate vs fixed planted poison configurations
e = pd.read_csv(RES / "s0e_realized_adversarial.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
ax = axes[0]
cfgs = ["split_79_21", "hug_threshold", "clean_like", "strong"]
labels = ["recall pinned\nat 0.79", "poisons hug\nthreshold", "clean-like\npoisons", "well\nseparated"]
for i, (rname, col, mk) in enumerate([("declared_correct", C[0], "o"), ("ceiling_too_low", C[3], "s")]):
    v = [e[(e.config == c) & (e.range == rname)].false_go.max() for c in cfgs]
    ax.scatter(np.arange(4) + (i - 0.5) * 0.25, v, color=col, marker=mk, s=28,
               label="count range correct" if i == 0 else "ceiling set too low")
ax.axhline(0.05, color=INK2, lw=1); ax.text(-0.45, 0.08, "δ = 0.05", fontsize=7.5, color=INK2)
ax.grid(False); ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("max false-GO rate"); ax.legend(fontsize=7, loc="center right")
ax.set_title("(a) Fixed-layout validity", fontsize=8.2, color=INK2, loc="left")
ax = axes[1]
for i, (pr, mk) in enumerate([(0.01, "o"), (0.05, "s"), (0.2, "^")]):
    s = e[(e.config == "strong") & (e.range == "declared_correct") & (e.prevalence == pr)].sort_values("k")
    ax.plot(s.k, s.p_go, color=C[i], marker=mk, ms=5)
    last = s.iloc[-1]
    ax.annotate(
        f"π = {pr:.0%}",
        (last.k, last.p_go),
        xytext=(-7, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color=C[i],
    )
ax.set_xscale("log"); ax.set_xlabel("trusted clean members k"); ax.set_ylabel("P(GO), recall ≈ 1")
ax.set_ylim(-0.02, 1.11)
ax.grid(False); ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_title("(b) Power vs. trusted members", fontsize=8.2, color=INK2, loc="left")
save(fig, "fig6_realized_adversarial")
