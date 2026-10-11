"""S0-b: price of label-free assurance.
(1) Minimal trusted calibration size k* for which the certificate issues GO with
    power >= 0.8 when the true TPR exceeds the target by a margin, versus the
    first-order prediction and the two-point (Le Cam) lower bound.
(2) Prevalence-ceiling tolerance: with k, n -> large, GO is possible only if
    pi_max / pi <= (TPR - alpha) / (tau - alpha).
Synthetic binormal scores only.
"""
import sys, pathlib, csv, math
import numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import conformal_rank, certify_counts, required_trusted_size, lecam_k_lower_bound_test
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _chunk import parse, mine, cell_rng, write
ALPHA, TAU, DELTA, R = 0.05, 0.8, 0.05, 3000
Z = stats.norm.isf(ALPHA)

def p_go(rng, k, n, pi, mu, pi_min=None, pi_max=None, eta=0.0):
    pi_min = pi if pi_min is None else pi_min
    pi_max = pi if pi_max is None else pi_max
    j = conformal_rank(k, ALPHA)
    U = rng.beta(k - j + 1, j, size=R)
    t = stats.norm.isf(U)
    tpr = stats.norm.sf(t - mu)
    K = rng.binomial(n, (1 - pi) * U + pi * tpr)
    c = certify_counts(k, j, K, n, alpha=ALPHA, tau=TAU, pi_min=pi_min, pi_max=pi_max,
                       delta=DELTA, eta=eta, alpha_cap=1.0)
    return float((c["decision"] == 1).mean())

def kstar(rng, pi, margin, n):
    mu = Z - stats.norm.isf(TAU + margin)          # TPR at the nominal threshold = tau + margin
    ks = np.unique(np.round(np.geomspace(20, 3e6, 46)).astype(int))
    prev_k, prev_p = None, None
    for k in ks:
        p = p_go(rng, int(k), n, pi, mu)
        if p >= 0.8:
            if prev_k is None:
                return float(k)
            # log-linear interpolation
            w = (0.8 - prev_p) / max(p - prev_p, 1e-9)
            return float(math.exp(math.log(prev_k) + w * (math.log(k) - math.log(prev_k))))
        prev_k, prev_p = k, p
    return float("nan")

def roc_slope_and_tail_ok(margin):
    """Binormal: ROC slope at the nominal threshold equals the likelihood ratio
    f1/f0 at z_alpha, which is also its minimum over the tail {s > z_alpha};
    Theorem 4's construction needs f1/f0 >= margin/alpha on that tail."""
    mu = Z - stats.norm.isf(TAU + margin)
    lr = math.exp(mu * Z - mu * mu / 2)
    return lr, lr >= margin / ALPHA

if __name__ == "__main__":
    a = parse(["all", "kstar", "ceiling"])
    if a.what in ("all", "kstar"):
        rows = []
        grid = [(pi, m) for pi in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4) for m in (0.05, 0.10, 0.15)]
        n = 10**8
        for cell, (pi, margin) in enumerate(grid):
            if not mine(cell, a):
                continue
            rng = cell_rng(11, cell)
            ke = kstar(rng, pi, margin, n)
            rho, ok = roc_slope_and_tail_ok(margin)
            rows.append(dict(cell=cell, n=n, pi=pi, margin=margin, k_star_empirical=ke,
                             k_star_first_order=required_trusted_size(ALPHA, pi, margin, DELTA, 0.8,
                                                                      roc_slope=rho),
                             roc_slope=rho, tail_condition_holds=ok,
                             k_lower_bound_test=lecam_k_lower_bound_test(ALPHA, pi, margin, DELTA, 0.8)
                             if ok else float("nan")))
            print(rows[-1], flush=True)
        print("kstar ->", write("s0b_kstar", rows, a))
    if a.what in ("all", "ceiling"):
        rows2 = []
        grid = [(tp, r) for tp in (0.85, 0.95, 0.99) for r in np.round(np.arange(1.0, 1.61, 0.05), 2)]
        for cell, (tpr_true, ratio) in enumerate(grid):
            if not mine(cell, a):
                continue
            rng = cell_rng(111, cell)
            mu = Z - stats.norm.isf(tpr_true)
            pred = (tpr_true - ALPHA) / (TAU - ALPHA)
            pi = 0.1
            pg = p_go(rng, 10**7, 10**8, pi, mu, pi_min=pi, pi_max=pi * ratio)
            rows2.append(dict(cell=cell, tpr_true=tpr_true, ceiling_ratio=ratio,
                              predicted_max_ratio=pred, p_go=pg))
        print("ceiling ->", write("s0b_ceiling", rows2, a))
