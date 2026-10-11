"""S0-d: poisoned samples leaking into the trusted calibration set.
Prediction: if TPR(t) >= FPR(t) at the operating threshold, contamination inflates
the calibration tail mass, so the FPR upper bound and hence TPR_lo stay valid
(no false GO), while TPR_hi can become too low (more false NO-GO / abstention).
Synthetic binormal scores; raw calibration samples."""
import sys, pathlib, csv, math, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import conformal_rank, certify_counts

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _chunk import parse, mine, cell_rng, write
ALPHA, DELTA, R = 0.05, 0.05, 4000
a = parse()
rows = []
for cell, (pi, mu, k, eps, tau) in enumerate(itertools.product([0.02, 0.05, 0.1, 0.2], [1.5, 2.5, 3.5],
                                             [120, 2000], [0.0, 0.03, 0.1, 0.2], [0.5, 0.8])):
    if not mine(cell, a):
        continue
    rng = cell_rng(5, cell)
    n = 50000
    j = conformal_rank(k, ALPHA)
    nbad = rng.binomial(k, eps, size=R)
    # j-th order statistic of a contaminated calibration sample
    t = np.empty(R)
    for r in range(R):
        cal = np.r_[rng.normal(size=k - nbad[r]), rng.normal(mu, 1, size=nbad[r])]
        t[r] = np.partition(cal, j - 1)[j - 1]
    fpr = stats.norm.sf(t); tpr = stats.norm.sf(t - mu)
    K = rng.binomial(n, (1 - pi) * fpr + pi * tpr)
    c = certify_counts(k, j, K, n, alpha=ALPHA, tau=tau, pi_min=pi, pi_max=pi, delta=DELTA,
                       alpha_cap=0.10)
    go, nogo = c["decision"] == 1, c["decision"] == -1
    rows.append(dict(cell=cell, pi=pi, mu=mu, k=k, eps=eps, tau=tau, mean_tpr=float(tpr.mean()),
                     mean_fpr=float(fpr.mean()),
                     lower_bound_violation=float(np.mean(c["tpr_lo"] > tpr)),
                     upper_bound_violation=float(np.mean(c["tpr_hi"] < tpr)),
                     p_go=float(go.mean()), false_go=float((go & ((tpr < tau) | (fpr > 0.10))).mean()),
                     false_nogo=float((nogo & (tpr >= tau)).mean())))
print(len(rows), "rows ->", write("s0d_contamination", rows, a))
