"""Compare two independent S0 runs (e.g. cloud vs local machine) on every headline
quantity.  The runs use different random streams, so agreement is judged against
Monte-Carlo error, not bit-identity.
Usage: python sims/compare_runs.py <dir_A> <dir_B> <label_A> <label_B> <out.md>"""
import sys, math
import numpy as np, pandas as pd

A, B, la, lb, out = sys.argv[1:6]
rows = []

def add(name, va, vb, tol, kind="abs"):
    if kind == "abs":
        diff = abs(va - vb); ok = diff <= tol
        rows.append((name, f"{va:.4g}", f"{vb:.4g}", f"|Δ| = {diff:.3g}", f"≤ {tol:.3g}", "一致" if ok else "**偏差**"))
    elif kind == "ratio":
        r = vb / va; ok = abs(math.log(r)) <= math.log(1 + tol)
        rows.append((name, f"{va:.4g}", f"{vb:.4g}", f"比值 {r:.3f}", f"±{tol:.0%}", "一致" if ok else "**偏差**"))
    elif kind == "both_below":
        ok = (va <= tol) and (vb <= tol)
        rows.append((name, f"{va:.4g}", f"{vb:.4g}", "两者均需 ≤ 阈值", f"≤ {tol:.3g}", "一致" if ok else "**偏差**"))
    elif kind == "equal":
        ok = va == vb
        rows.append((name, str(va), str(vb), "需相同", "—", "一致" if ok else "**偏差**"))

def se(p, R):
    return math.sqrt(max(p * (1 - p), 1e-4) / R)

# ---------- S0-a
a = pd.read_csv(f"{A}/s0a_main.csv"); b = pd.read_csv(f"{B}/s0a_main.csv")
for rng in ["declared_exact", "declared_correct"]:
    for tau in [0.8, 0.9]:
        add(f"S0-a 最大错误 GO（{rng}, τ={tau}）",
            a[(a.range == rng) & (a.tau == tau)].far_pop.max(),
            b[(b.range == rng) & (b.tau == tau)].far_pop.max(), 0.05, "both_below")
pa = a[(a.tau == 0.8) & (a.range == "declared_exact")]; pb = b[(b.tau == 0.8) & (b.range == "declared_exact")]
add("S0-a 点估计对照：最大错误 GO", pa.plugin_far_pop.max(), pb.plugin_far_pop.max(), 0.04)
add("S0-a 点估计对照：错误 GO>5% 的单元比例", (pa.plugin_far_pop > 0.05).mean(), (pb.plugin_far_pop > 0.05).mean(), 0.03)
ma = a[(a.tau == 0.8) & (a.range == "declared_misspecified")]; mb = b[(b.tau == 0.8) & (b.range == "declared_misspecified")]
add("S0-a 占比上限设低：错误 GO>5% 的单元比例", (ma.far_pop > 0.05).mean(), (mb.far_pop > 0.05).mean(), 0.03)
key = ["family", "pi", "mu", "k", "n", "tau", "range"]
m = a.merge(b, on=key, suffixes=("_a", "_b"))
d = (m.p_go_a - m.p_go_b).abs() / np.sqrt([se(p, 4000) ** 2 * 2 for p in (m.p_go_a + m.p_go_b) / 2])
add("S0-a 全部单元 P(GO) 的差 / 蒙特卡洛标准误：99% 分位", float(np.quantile(d, 0.99)), 0.0, 3.5, "abs")
rows[-1] = (rows[-1][0], f"{float(np.quantile(d, 0.99)):.3g}", "—", "z 分数", "≤ 3.5", "一致" if np.quantile(d, 0.99) <= 3.5 else "**偏差**")
ta = pd.read_csv(f"{A}/s0a_transport.csv"); tb = pd.read_csv(f"{B}/s0a_transport.csv")
for pi, s in [(0.01, 0.1), (0.01, 0.2), (0.05, 0.1)]:
    x = lambda t: float(t[(t.mu == 1.5) & (t.k == 2000) & (t.pi == pi) & (t["shift"] == s) & (t["mode"] == "fresh_eta0")]["far_tau0.5"].iloc[0])
    add(f"S0-a 迁移：新样本校准错误 GO（π={pi}, s={s}）", x(ta), x(tb), 4 * se(x(ta), 4000) + 0.01)
mx = lambda t: float(t[t["mode"] == "trusted_members"]["far_tau0.5"].max())
add("S0-a 迁移：可信成员校准最大错误 GO", mx(ta), mx(tb), 0.05, "both_below")
# ---------- S0-b
ka = pd.read_csv(f"{A}/s0b_kstar.csv"); kb = pd.read_csv(f"{B}/s0b_kstar.csv")
for _, r in ka.iterrows():
    rb = kb[(kb.pi == r.pi) & (kb.margin == r.margin)].iloc[0]
    add(f"S0-b k*（π={r.pi}, 裕量={r.margin}）", r.k_star_empirical, rb.k_star_empirical, 0.15, "ratio")
for lab, df in ((la, ka), (lb, kb)):
    pass
sl = lambda df: np.polyfit(np.log(df[(df.margin == 0.1) & (df.pi <= 0.1)].pi),
                           np.log(df[(df.margin == 0.1) & (df.pi <= 0.1)].k_star_empirical), 1)[0]
add("S0-b k* 对 π 的对数斜率（裕量 0.1）", sl(ka), sl(kb), 0.08)
ca = pd.read_csv(f"{A}/s0b_ceiling.csv"); cb = pd.read_csv(f"{B}/s0b_ceiling.csv")
for tp in (0.85, 0.95, 0.99):
    f = lambda c: float(c[(c.tpr_true == tp) & (c.p_go < 0.5)].ceiling_ratio.min())
    add(f"S0-b GO 消失的上限比（真实 TPR={tp}）", f(ca), f(cb), 0.051)
# ---------- S0-c
sa = pd.read_csv(f"{A}/s0c_direction_budget.csv"); sb = pd.read_csv(f"{B}/s0c_direction_budget.csv")
for sp in ("isotropic", "anisotropic"):
    ea = float((sa[sa.spectrum == sp].tpr_pred - sa[sa.spectrum == sp].tpr_sim).mean())
    eb = float((sb[sb.spectrum == sp].tpr_pred - sb[sb.spectrum == sp].tpr_sim).mean())
    add(f"S0-c 预测−模拟平均偏差（{sp}）", ea, eb, 0.015)
mm = sa.merge(sb, on=["spectrum", "d", "pi", "pi_r", "m"], suffixes=("_a", "_b"))
mad = float((mm.tpr_sim_a - mm.tpr_sim_b).abs().mean())
rows.append(("S0-c 两次模拟 TPR 的平均绝对差（同一单元）", f"{mad:.3g}", "—", "平均 |Δ|", "≤ 0.03",
             "一致" if mad <= 0.03 else "**偏差**"))
# ---------- S0-d
da = pd.read_csv(f"{A}/s0d_contamination.csv"); db = pd.read_csv(f"{B}/s0d_contamination.csv")
for eps in (0.03, 0.1, 0.2):
    add(f"S0-d 下界违约率最大值（ε={eps}）", da[da.eps == eps].lower_bound_violation.max(),
        db[db.eps == eps].lower_bound_violation.max(), 0.05, "both_below")
add("S0-d ε>0 时上界违约率最大值", da[da.eps > 0].upper_bound_violation.max(),
    db[db.eps > 0].upper_bound_violation.max(), 0.01)
# ---------- S0-e
ea_ = pd.read_csv(f"{A}/s0e_realized_adversarial.csv"); eb_ = pd.read_csv(f"{B}/s0e_realized_adversarial.csv")
for cfg in ("split_79_21", "hug_threshold", "clean_like", "strong"):
    add(f"S0-e 计数区间正确时最大错误 GO（{cfg}）",
        ea_[(ea_.config == cfg) & (ea_.range == "declared_correct")].false_go.max(),
        eb_[(eb_.config == cfg) & (eb_.range == "declared_correct")].false_go.max(), 0.05, "both_below")
add("S0-e 上限设低时最大错误 GO（split_79_21）",
    ea_[(ea_.config == "split_79_21") & (ea_.range == "ceiling_too_low")].false_go.max(),
    eb_[(eb_.config == "split_79_21") & (eb_.range == "ceiling_too_low")].false_go.max(), 0.02)
for pr, k in [(0.05, 5000), (0.2, 1000)]:
    g = lambda e: float(e[(e.config == "strong") & (e.range == "declared_correct") & (e.prevalence == pr) & (e.k == k)].p_go.iloc[0])
    add(f"S0-e 明显可分时 P(GO)（占比 {pr:.0%}, k={k}）", g(ea_), g(eb_), 4 * se(g(ea_), 3000) + 0.01)

n_ok = sum(r[-1] == "一致" for r in rows)
with open(out, "w") as f:
    f.write(f"# S0 两次独立运行的一致性核对：{la} vs {lb}\n\n")
    f.write(f"两次运行使用不同的随机数流，因此按蒙特卡洛误差判断是否一致，而不要求逐位相同。共 {len(rows)} 项，一致 {n_ok} 项。\n\n")
    f.write(f"| 指标 | {la} | {lb} | 差异 | 容差 | 判定 |\n|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(r) + " |\n")
print(f"{n_ok}/{len(rows)} consistent ->", out)
