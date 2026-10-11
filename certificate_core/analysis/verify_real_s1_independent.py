"""Independent re-check of the real S1 hand-off table (score table only; no models,
no checkpoints).  Uses fresh random draws and its own aggregation code, and adds two
theory-linked analyses:
  (1) tolerable ceiling ratio r* = (K - M_hi) / (tau * N_p): the largest declared
      N_max / N_p at which the certificate would still say GO (Corollary 1, realized
      form: r* <= recall / tau, so r* <= 1.25 even for a perfect detector at tau=0.8);
  (2) certified recall as a continuous function of the declared ceiling,
      recall_lo(rho) = (K - M_hi) / (rho * N_p), instead of a binary GO at tau.
Output: results_real_s1/independent_check/
"""
import sys, pathlib, json, math
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import conformal_rank, certify_realized_counts

ROOT = pathlib.Path(__file__).resolve().parents[1]
IN = ROOT / "results_real_s1" / "bloodmnist_s1_scores.csv"
import os
ALPHA, TAU, R, SEED = 0.05, 0.8, 400, 20261008
DELTA = float(os.environ.get("CERT_DELTA", 0.025))   # prereg: delta/F with F=2 folds
OUT = ROOT / "results_real_s1" / f"independent_check_delta{DELTA:g}"
OUT.mkdir(exist_ok=True)
RHOS = [1.0, 1.1, 1.2, 1.5, 2.0, 4.0]

df = pd.read_csv(IN)
rng = np.random.default_rng(SEED)
counts, rows = [], []
for (vid, fold), g in df.groupby(["victim_id", "fold"]):
    view = "paired_hidden64_drift" if "paired" in vid else "suspect_hidden64"
    seed = int(vid.rsplit("seed", 1)[1])
    s = g.score.to_numpy(float); role = g.role.to_numpy(); p = g.is_poison.to_numpy(int)
    # one global random tie-breaking order (ranks)
    order = np.lexsort((rng.random(s.size), s)); rk = np.empty(s.size); rk[order] = np.arange(s.size)
    clean_tr = np.where((role == "train") & (p == 0))[0]
    pois = np.where((role == "train") & (p == 1))[0]
    fresh = np.where(role == "fresh")[0]
    n_p = len(pois)
    counts.append(dict(victim_id=vid, view=view, seed=seed, fold=fold, clean_members=len(clean_tr),
                       poisons=n_p, fresh=len(fresh)))
    for k in (120, 500):
        if k >= len(clean_tr):
            continue
        j = conformal_rank(k, ALPHA)
        Ks, kps, fm, ff = [], [], [], []
        is_clean_tr = np.zeros(s.size, bool); is_clean_tr[clean_tr] = True
        for r in range(R):
            tr = rng.choice(clean_tr, k, replace=False)
            cand = (role == "train"); cand = cand.copy(); cand[tr] = False
            t = np.sort(rk[tr])[j - 1]
            above = rk > t
            flagged = cand & above
            Ks.append(int(flagged.sum())); kps.append(int((flagged & (p == 1)).sum()))
            rest = is_clean_tr.copy(); rest[tr] = False
            fm.append(float(above[rest].mean())); ff.append(float(above[fresh].mean()))
        Ks = np.array(Ks); kps = np.array(kps); rec = kps / n_p
        n = int((role == "train").sum()) - k
        block = dict(view=[view] * R, seed=[seed] * R, fold=[fold] * R, k=[k] * R, rep=list(range(R)),
                     recall=rec, caught=kps, member_fpr=fm, fresh_fpr=ff)
        for rho in RHOS:
            c = certify_realized_counts(k, j, Ks, n, tau=TAU, n_poison_min=n_p // 2,
                                        n_poison_max=int(math.ceil(rho * n_p)), delta=DELTA,
                                        alpha_cap=0.10)
            go = (c["decision"] == 1)
            block[f"go_{rho}"] = go.astype(int)
            block[f"false_go_{rho}"] = (go & (rec < TAU)).astype(int)
            block[f"recall_lo_{rho}"] = np.asarray(c["recall_lo"], float)
            if rho == 1.0:
                block["caught_lo"] = np.asarray(c["flagged_poisons_lo"], float)
        block["r_star"] = block["caught_lo"] / (TAU * n_p)
        rows.append(pd.DataFrame(block))

C = pd.DataFrame(counts); D = pd.concat(rows, ignore_index=True)
C.to_csv(OUT / "counts.csv", index=False); D.to_csv(OUT / "draws.csv", index=False)

summ = []
for (view, k), g in D.groupby(["view", "k"]):
    d = dict(view=view, k=k, draws=len(g), mean_recall=g.recall.mean(),
             caught_lo_over_true=(g.caught_lo / g.caught).mean(),
             member_minus_fresh=(g.member_fpr - g.fresh_fpr).mean(),
             r_star_median=g.r_star.median(), r_star_p05=g.r_star.quantile(0.05),
             r_star_p95=g.r_star.quantile(0.95), r_star_theory_cap=g.recall.mean() / TAU)
    for rho in RHOS:
        d[f"go_{rho}"] = g[f"go_{rho}"].mean()
        d[f"false_go_{rho}"] = g[f"false_go_{rho}"].mean()
        d[f"certified_recall_{rho}"] = g[f"recall_lo_{rho}"].mean()
    summ.append(d)
S = pd.DataFrame(summ); S.to_csv(OUT / "summary.csv", index=False)

pool = D.groupby("k").agg(go_exact=("go_1.0", "mean"), go_12=("go_1.2", "mean"),
                          false_go_exact=("false_go_1.0", "mean"),
                          false_go_any=("false_go_1.0", "max")).reset_index()
# Prop 3 on real numbers: bias of an i.i.d. fresh-calibrated estimate = (1-pi) b / pi
b = (D[D.k == 500].assign(gap=lambda x: x.member_fpr - x.fresh_fpr).groupby("view").gap.mean())
pi_real = C.poisons.sum() / (C.poisons.sum() + C.clean_members.sum())
prop3 = {view: {f"pi={pi:.2f}": (1 - pi) * float(bv) / pi for pi in (pi_real, 0.10, 0.05, 0.01)}
         for view, bv in b.items()}
res = dict(delta_per_fold=DELTA, input_rows=len(df), victims_folds=len(C), prevalence_audited=pi_real,
           clean_members_per_fold=[int(C.clean_members.min()), int(C.clean_members.max())],
           poisons_per_fold=[int(C.poisons.min()), int(C.poisons.max())],
           fresh_per_fold=[int(C.fresh.min()), int(C.fresh.max())],
           pooled=pool.to_dict(orient="records"), prop3_recall_bias_if_fresh_calibrated=prop3)
json.dump(res, open(OUT / "independent_check.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(json.dumps(res, indent=1))
print(S[["view", "k", "mean_recall", "go_1.0", "go_1.1", "go_1.2", "false_go_1.0", "caught_lo_over_true",
         "r_star_median", "r_star_p05", "r_star_p95", "r_star_theory_cap",
         "certified_recall_1.2", "certified_recall_1.5", "certified_recall_2.0"]].round(4).to_string(index=False))
