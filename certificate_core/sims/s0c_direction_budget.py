"""S0-c: why a small trusted reference destroys a mean-shift screening rule.
Synthetic Gaussian features only: clean ~ N(0, Sigma), poisoned ~ N(Delta, Sigma).
Rule (covariance-free, cross-fitted): direction v = mean(candidates, other fold) -
mean(reference); score = <x - mean(reference), v/|v|>; threshold = split-conformal
rank on a separate trusted calibration set.  Prediction (isotropic Sigma = I):
  cos^2 ~= (pi r)^2 / ((pi r)^2 + tr(Sigma) (1/m + 2/n)),   TPR ~= Phi(r cos - z_alpha),
so the transition sits at m* = tr(Sigma) / (pi r)^2 (with n >> m).  With an
anisotropic Sigma, the noise along Delta and tr(Sigma) separate, and the effective
dimension tr(Sigma)/sigma_Delta^2 replaces d.
"""
import sys, pathlib, csv, math, itertools
import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _chunk import parse, mine, cell_rng, write
rng = None  # set per cell
ALPHA = 0.05
Z = stats.norm.isf(ALPHA)
KCAL = 80
J = math.ceil((1 - ALPHA) * (KCAL + 1))

def one_run(rng, d, m, n, pi, r, spectrum, R=300):
    if spectrum == "isotropic":
        sd = np.ones(d)
    else:   # anisotropic: geometric spectrum, poison shift along the smallest-variance axis
        sd = np.sqrt(np.geomspace(4.0, 0.25, d))
    delta = np.zeros(d)
    axis = d - 1 if spectrum != "isotropic" else 0
    delta[axis] = r * sd[axis]           # r measured in noise s.d. along Delta
    tprs = []
    for _ in range(R):
        ref = rng.normal(size=(m, d)) * sd
        cal = rng.normal(size=(KCAL, d)) * sd
        npz = rng.binomial(n, pi)
        lab = np.zeros(n, bool); lab[:npz] = True; rng.shuffle(lab)
        cand = rng.normal(size=(n, d)) * sd + lab[:, None] * delta
        folds = np.arange(n) % 2
        mu_ref = ref.mean(0)
        hits = tot = 0
        for f in (0, 1):
            v = cand[folds != f].mean(0) - mu_ref
            u = v / np.linalg.norm(v)
            s_cal = (cal - mu_ref) @ u
            t = np.sort(s_cal)[J - 1]
            s = (cand[folds == f] - mu_ref) @ u
            pl = lab[folds == f]
            hits += int((s[pl] > t).sum()); tot += int(pl.sum())
        tprs.append(hits / max(tot, 1))
    return float(np.mean(tprs))

def predict(d, m, n, pi, r, spectrum):
    sd2 = np.ones(d) if spectrum == "isotropic" else np.geomspace(4.0, 0.25, d)
    sig_delta2 = sd2[0] if spectrum == "isotropic" else sd2[-1]
    # signal along Delta = pi * r * sigma_Delta ; direction noise = tr(Sigma)(1/m + 2/n)
    sig = (pi * r) ** 2 * sig_delta2
    cos2 = sig / (sig + sd2.sum() * (1 / m + 2 / n))
    # score noise along u is ~ sigma_Delta^2 cos^2 + (rest) ; use isotropic-in-rest approx
    rest = (sd2.sum() - sig_delta2) / (d - 1) if d > 1 else 0.0
    noise_sd = math.sqrt(sig_delta2 * cos2 + rest * (1 - cos2))
    return float(stats.norm.sf(Z - r * math.sqrt(sig_delta2) * math.sqrt(cos2) / noise_sd))

if __name__ == "__main__":
    a = parse()
    rows = []
    ms = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]
    combos = [c for c in itertools.product(["isotropic", "anisotropic"], [8, 16, 32, 64],
                                           [0.05, 0.1, 0.2], [0.5, 1.0, 2.0]) if c[3] / c[2] <= 12]
    for cell, (spectrum, d, pi, pir) in enumerate(combos):
        if not mine(cell, a):
            continue
        rng = cell_rng(3, cell)
        r = pir / pi
        n = 1200
        for m in ms:
            tpr = one_run(rng, d, m, n, pi, r, spectrum, R=120)
            sd2 = np.ones(d) if spectrum == "isotropic" else np.geomspace(4.0, 0.25, d)
            sdel = sd2[0] if spectrum == "isotropic" else sd2[-1]
            m_star = sd2.sum() / ((pi * r) ** 2 * sdel)
            rows.append(dict(cell=cell, spectrum=spectrum, d=d, pi=pi, pi_r=pir, r=r, m=m, n=n,
                             tpr_sim=tpr, tpr_pred=predict(d, m, n, pi, r, spectrum),
                             m_star=m_star, m_over_mstar=m / m_star))
        print(spectrum, d, pi, pir, "done", flush=True)
    print("->", write("s0c_direction_budget", rows, a))
