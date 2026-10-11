"""S0-a: finite-sample validity, power and failure modes of the label-free
auditability certificate on synthetic score distributions (no models, no data
sets, no attacks).  Sampling uses exact sufficient statistics:
  U = FPR_F0(t) ~ Beta(k-j+1, j),  t = F0^{-1}(1-U),
  N_p ~ Bin(n, pi), K_p ~ Bin(N_p, TPR(t)), K_c ~ Bin(n-N_p, FPR_G0(t)).
"""
import sys, pathlib, json, itertools, time
import numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import conformal_rank, certify_counts
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _chunk import parse, mine, cell_rng, write
ALPHA, CAP, DELTA, R = 0.05, 0.10, 0.05, 4000

# ROC shapes (the certificate is rank-based, so only the ROC matters)
FAMILIES = {
    "binormal":         (stats.norm(), lambda mu: stats.norm(mu, 1)),
    "binormal_unequal": (stats.norm(), lambda mu: stats.norm(mu, 2)),
    "partial_40pct":    (stats.norm(), None),             # 40% of poisons clean-like
    "t3":               (stats.t(3),   lambda mu: stats.t(3, loc=mu)),
}

def tpr_of(fam, mu, t):
    if fam == "partial_40pct":
        return 0.6 * stats.norm.sf(t - mu) + 0.4 * stats.norm.sf(t)
    return FAMILIES[fam][1](mu).sf(t)

def ranges(pi):
    return {"declared_correct": (pi / 2, min(2 * pi, 1.0)),
            "declared_exact": (pi, pi),
            "declared_misspecified": (pi / 4, pi / 2)}

def run_main(a):
    rows = []
    grid = itertools.product(FAMILIES, [0.01, 0.02, 0.05, 0.1, 0.2], [1.0, 2.0, 3.0, 4.0],
                             [32, 120, 500, 2000, 10000], [2000, 10000, 50000])
    for cell, (fam, pi, mu, k, n) in enumerate(grid):
        if not mine(cell, a):
            continue
        rng = cell_rng(7, cell)
        f0 = FAMILIES[fam][0]
        j = conformal_rank(k, ALPHA)
        U = rng.beta(k - j + 1, j, size=R)
        t = f0.isf(U)
        tpr = tpr_of(fam, mu, t)
        Np = rng.binomial(n, pi, size=R)
        Kp = rng.binomial(Np, tpr)
        Kc = rng.binomial(n - Np, U)
        K = Kp + Kc
        real_rec = np.where(Np > 0, Kp / np.maximum(Np, 1), np.nan)
        valid_real = Np > 0
        real_fpr = Kc / (n - Np)
        # plug-in baseline: oracle pi, nominal FPR = alpha, no uncertainty
        plug = (K / n - (1 - pi) * ALPHA) / pi
        plug_go = plug >= 0.8
        for tau in (0.8, 0.9):
            bad_pop = (tpr < tau) | (U > CAP)
            bad_real = np.where(valid_real, (np.nan_to_num(real_rec, nan=1.0) < tau) | (real_fpr > CAP), False)
            for rname, (pmin, pmax) in ranges(pi).items():
                c = certify_counts(k, j, K, n, alpha=ALPHA, tau=tau, pi_min=pmin, pi_max=pmax,
                                   delta=DELTA, alpha_cap=CAP)
                go, nogo = c["decision"] == 1, c["decision"] == -1
                cov = np.mean((c["tpr_lo"] <= tpr) & (tpr <= c["tpr_hi"]))
                rows.append(dict(cell=cell, family=fam, pi=pi, mu=mu, k=k, n=n, tau=tau, range=rname,
                                 mean_tpr=float(tpr.mean()), p_go=float(go.mean()),
                                 p_nogo=float(nogo.mean()), far_pop=float((go & bad_pop).mean()),
                                 far_real=float((go & bad_real)[valid_real].mean()) if valid_real.any() else np.nan,
                                 false_nogo=float((nogo & (tpr >= tau)).mean()),
                                 coverage=float(cov),
                                 mean_width=float((c["tpr_hi"] - c["tpr_lo"]).mean()),
                                 plugin_far_pop=float((plug_go & bad_pop).mean()) if tau == 0.8 else np.nan,
                                 plugin_go=float(plug_go.mean()) if tau == 0.8 else np.nan))
    return rows

def run_transport(a):
    """Clean candidates (training members) score higher than fresh trusted samples
    by a location shift s.  Compare fresh calibration (eta=0), fresh calibration
    with a declared tolerance, and calibration on trusted members."""
    rows = []
    z = stats.norm.isf(ALPHA)
    for cell, (pi, mu, s, k) in enumerate(itertools.product([0.01, 0.02, 0.05, 0.1, 0.2], [1.5, 2.0, 3.0],
                                          [0.0, 0.05, 0.1, 0.2], [120, 2000])):
        if not mine(cell, a):
            continue
        rng = cell_rng(70, cell)
        n = 50000
        j = conformal_rank(k, ALPHA)
        U = rng.beta(k - j + 1, j, size=R)
        eta_point = 1.5 * (stats.norm.sf(z - s) - ALPHA)       # declared pointwise tolerance
        eta_ks = 2 * stats.norm.cdf(s / 2) - 1                 # KS distance (conservative)
        for mode in ("fresh_eta0", "fresh_eta_point", "fresh_eta_ks", "trusted_members"):
            if mode == "trusted_members":
                t = stats.norm.isf(U) + s          # calibrate on the member law G0 = N(s,1)
            else:
                t = stats.norm.isf(U)              # calibrate on fresh law F0 = N(0,1)
            fpr_c = stats.norm.sf(t - s)           # true clean-candidate FPR
            tpr = stats.norm.sf(t - mu)
            Np = rng.binomial(n, pi, size=R)
            Kp = rng.binomial(Np, tpr); Kc = rng.binomial(n - Np, fpr_c)
            K = Kp + Kc
            eta = {"fresh_eta0": 0.0, "fresh_eta_point": eta_point, "fresh_eta_ks": eta_ks,
                   "trusted_members": 0.0}[mode]
            res = {}
            for tau in (0.5, 0.8):
                c = certify_counts(k, j, K, n, alpha=ALPHA, tau=tau, pi_min=pi, pi_max=pi,
                                   delta=DELTA, eta=eta, alpha_cap=CAP)
                go = c["decision"] == 1
                bad = (tpr < tau) | (fpr_c > CAP)
                res[f"p_go_tau{tau}"] = float(go.mean())
                res[f"far_tau{tau}"] = float((go & bad).mean())
            centre = (K / n - (1 - pi) * U) / pi    # point estimate that ignores transport
            rows.append(dict(cell=cell, pi=pi, mu=mu, shift=s, k=k, mode=mode, eta=eta, **res,
                             fpr_gap_at_t=float((fpr_c - (U if mode != "trusted_members" else fpr_c)).mean()),
                             mean_tpr=float(tpr.mean()),
                             tpr_bias_of_point_estimate=float((centre - tpr).mean()) if mode == "fresh_eta0" else np.nan,
                             coverage=float(np.mean((c["tpr_lo"] <= tpr) & (tpr <= c["tpr_hi"]))),
                             mean_true_fpr=float(fpr_c.mean())))
    return rows

if __name__ == "__main__":
    t0 = time.time()
    a = parse(["all", "main", "transport"])
    if a.what in ("all", "main"):
        print("main ->", write("s0a_main", run_main(a), a))
    if a.what in ("all", "transport"):
        print("transport ->", write("s0a_transport", run_transport(a), a))
    print(f"time={time.time()-t0:.1f}s")
