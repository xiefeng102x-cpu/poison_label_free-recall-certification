"""S0-e: realized-recall certificate against fixed (adversarially planted) poison
score configurations.  Finite population: one fixed set of clean scores per
configuration; the ONLY randomness is the defender's secret, uniformly random
choice of k trusted clean members.  Synthetic numbers only."""
import sys, pathlib, csv, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import conformal_rank, certify_realized_counts

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _chunk import parse, mine, cell_rng, write
ALPHA, DELTA, TAU, R = 0.05, 0.05, 0.8, 3000
N_CLEAN = 50000

def poison_config(rng, name, n_p, clean_sorted):
    q = lambda p: np.quantile(clean_sorted, p)
    if name == "split_79_21":            # realized recall pinned at 0.79 < tau
        hi = int(round(0.79 * n_p)); return np.r_[np.full(hi, q(1) + 50), np.full(n_p - hi, q(0) - 50)]
    if name == "hug_threshold":          # poisons just above the nominal 95% clean quantile
        return np.full(n_p, q(0.955))
    if name == "clean_like":             # indistinguishable from clean
        return rng.choice(clean_sorted, n_p, replace=True)
    if name == "strong":                 # well separated (power check)
        return q(0.5) + 4.0 + rng.normal(size=n_p)
    raise ValueError(name)

a = parse()
rows = []
for cell, (cfg, frac, k) in enumerate(itertools.product(["split_79_21", "hug_threshold", "clean_like", "strong"],
                                      [0.01, 0.05, 0.2], [120, 1000, 5000])):
    if not mine(cell, a):
        continue
    rng = cell_rng(17, cell)
    n_p = int(round(frac * N_CLEAN / (1 - frac)))
    clean = np.sort(rng.normal(size=N_CLEAN))
    pois = np.sort(poison_config(rng, cfg, n_p, clean))
    j = conformal_rank(k, ALPHA)
    n = N_CLEAN - k + n_p
    K = np.empty(R, int); Kp = np.empty(R, int)
    for r in range(R):
        idx = rng.choice(N_CLEAN, k, replace=False)
        t = np.sort(clean[idx])[j - 1]
        clean_above = N_CLEAN - np.searchsorted(clean, t, side="right") - (k - j)
        kp = n_p - np.searchsorted(pois, t, side="right")
        K[r] = clean_above + kp; Kp[r] = kp
    rec = Kp / n_p
    for rng_name, (nmin, nmax) in {"declared_correct": (n_p // 2, n_p),
                                   "ceiling_too_low": (n_p // 4, n_p // 2)}.items():
        c = certify_realized_counts(k, j, K, n, tau=TAU, n_poison_min=nmin, n_poison_max=nmax,
                                    delta=DELTA)
        go, nogo = c["decision"] == 1, c["decision"] == -1
        rows.append(dict(cell=cell, config=cfg, prevalence=frac, n_poison=n_p, k=k, range=rng_name,
                         mean_realized_recall=float(rec.mean()),
                         p_go=float(go.mean()), false_go=float((go & (rec < TAU)).mean()),
                         false_nogo=float((nogo & (rec >= TAU)).mean()),
                         lower_bound_violation=float((c["recall_lo"] > rec).mean()),
                         mean_recall_lo=float(c["recall_lo"].mean())))
        print(rows[-1], flush=True)
print("->", write("s0e_realized_adversarial", rows, a))
