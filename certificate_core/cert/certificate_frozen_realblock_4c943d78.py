"""Label-free auditability certificate for sample-level poison screening.

The certificate wraps ANY detector that outputs a real-valued suspicion score
(higher = more suspicious).  It uses only
  * k trusted clean calibration scores  (ideally trusted *members*, see transport),
  * n unlabeled candidate scores        (training samples under audit),
  * a declared prevalence range          pi in [pi_min, pi_max],
and never uses poison labels.

Model (per fold of a cross-fitted detector, conditional on the fitted score map):
  calibration  c_1..c_k  iid F0                      (trusted clean)
  candidates   x_1..x_n  iid (1-pi) G0 + pi F1       (G0 = clean-candidate law)
  transport    |FPR_G0(t) - FPR_F0(t)| <= eta at the operating threshold.

Threshold: t = c_(j), j = ceil((1-alpha)(k+1))   (split-conformal rank rule).
Exact law: U = FPR_F0(t) ~ Beta(k-j+1, j) for continuous F0 (distribution free).
Candidate tail count K = #{x_i > t} | t ~ Binomial(n, q(t)),
  q(t) = (1-pi) FPR_G0(t) + pi TPR(t).
Identity: TPR(t) = (q(t) - (1-pi) FPR_G0(t)) / pi.

Only standard numpy/scipy are used.  No model, data set, or attack is touched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# primitive pieces
# ----------------------------------------------------------------------------
def conformal_rank(k: int, alpha: float) -> int:
    """Rank j of the split-conformal threshold c_(j) (1-based)."""
    j = math.ceil((1.0 - alpha) * (k + 1))
    if j > k:
        raise ValueError(
            f"k={k} trusted scores are too few for alpha={alpha}: need k >= {math.ceil(1/alpha) - 1}")
    return j


def fpr_beta_bounds(k, j, delta1):
    """Two-sided exact (1-delta1) interval for FPR_F0(c_(j)) ~ Beta(k-j+1, j)."""
    a = np.asarray(k) - np.asarray(j) + 1
    b = np.asarray(j)
    lo = stats.beta.ppf(delta1 / 2.0, a, b)
    hi = stats.beta.ppf(1.0 - delta1 / 2.0, a, b)
    return lo, hi


def clopper_pearson(K, n, delta2):
    """Two-sided exact (1-delta2) Clopper-Pearson interval for a binomial p."""
    K = np.asarray(K, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(K > 0, stats.beta.ppf(delta2 / 2.0, K, n - K + 1), 0.0)
        hi = np.where(K < n, stats.beta.ppf(1.0 - delta2 / 2.0, K + 1, n - K), 1.0)
    return lo, hi


def _tpr(pi, q, f):
    """TPR implied by candidate tail q, clean-candidate FPR f and prevalence pi."""
    return (q - (1.0 - pi) * f) / pi


# ----------------------------------------------------------------------------
# certificate from sufficient statistics (vectorised)
# ----------------------------------------------------------------------------
def certify_counts(k, j, K, n, *, alpha, tau, pi_min, pi_max,
                   delta=0.05, eta=0.0, alpha_cap=None, split=0.5, sided="one"):
    """Certificate from (k, j, K, n).  All arguments may be numpy arrays.

    Returns dict of arrays: fpr_lo/hi (clean-candidate FPR bounds incl. eta),
    q_lo/hi, tpr_lo/hi, flagged_poison_frac_lo (no prevalence assumption),
    residual_poison_frac_hi (uses pi_max), decision codes
      1 = GO (TPR_lo >= tau and FPR_hi <= alpha_cap)
     -1 = NO-GO (TPR_hi < tau: certified failure for every pi in range)
      0 = UNDETERMINED (abstain)
    sided="one" (default): every one-sided statement (TPR >= tpr_lo, FPR <= fpr_hi,
      TPR <= tpr_hi, ...) holds with probability >= 1 - delta on its own, so
      P(false GO) <= delta and P(false NO-GO) <= delta; [tpr_lo, tpr_hi] then
      covers with probability >= 1 - 2 delta.
    sided="two": all statements hold jointly with probability >= 1 - delta.
    Validity requires pi in [pi_min, pi_max] and the transport bound eta.
    """
    if alpha_cap is None:
        alpha_cap = 2.0 * alpha
    if not (0 < pi_min <= pi_max <= 1):
        raise ValueError("need 0 < pi_min <= pi_max <= 1")
    d1, d2 = delta * split, delta * (1.0 - split)
    if sided == "two":
        u_lo, u_hi = fpr_beta_bounds(k, j, d1)
        q_lo, q_hi = clopper_pearson(K, n, d2)
    elif sided == "one":
        u_lo, u_hi = fpr_beta_bounds(k, j, 2 * d1)
        q_lo, q_hi = clopper_pearson(K, n, 2 * d2)
    else:
        raise ValueError("sided must be 'one' or 'two'")
    f_lo = np.clip(u_lo - eta, 0.0, 1.0)
    f_hi = np.clip(u_hi + eta, 0.0, 1.0)
    # TPR(pi) = (q - f)/pi + f is monotone in pi -> extremes at the endpoints
    lo_cands = np.stack([_tpr(pi_min, q_lo, f_hi), _tpr(pi_max, q_lo, f_hi)])
    hi_cands = np.stack([_tpr(pi_min, q_hi, f_lo), _tpr(pi_max, q_hi, f_lo)])
    tpr_lo = np.clip(lo_cands.min(axis=0), 0.0, 1.0)
    tpr_hi = np.clip(hi_cands.max(axis=0), 0.0, 1.0)
    flagged_lo = np.clip(q_lo - f_hi, 0.0, 1.0)                      # pi*TPR >= q - f
    residual_hi = np.clip(pi_max - q_lo + (1.0 - pi_max) * f_hi, 0.0, 1.0)  # pi*(1-TPR)
    go = (tpr_lo >= tau) & (f_hi <= alpha_cap)
    nogo = (tpr_hi < tau)
    decision = np.where(go, 1, np.where(nogo, -1, 0))
    return dict(fpr_lo=f_lo, fpr_hi=f_hi, q_lo=q_lo, q_hi=q_hi, tpr_lo=tpr_lo,
                tpr_hi=tpr_hi, flagged_poison_frac_lo=flagged_lo,
                residual_poison_frac_hi=residual_hi, decision=decision)


# ----------------------------------------------------------------------------
# certificate from raw score arrays (what a defender would call)
# ----------------------------------------------------------------------------
@dataclass
class Certificate:
    threshold: float
    k: int
    j: int
    n: int
    flagged: int
    fpr_lo: float
    fpr_hi: float
    tpr_lo: float
    tpr_hi: float
    flagged_poison_frac_lo: float
    residual_poison_frac_hi: float
    decision: str
    alpha: float
    tau: float
    pi_min: float
    pi_max: float
    delta: float
    eta: float
    sided: str = "one"

    def as_dict(self):
        return asdict(self)


def certify(cal_scores, cand_scores, *, alpha=0.05, tau=0.8, pi_min, pi_max,
            delta=0.05, eta=0.0, alpha_cap=None, sided="one") -> Certificate:
    """Audit one fold.  cal_scores: trusted clean scores; cand_scores: candidates."""
    cal = np.asarray(cal_scores, dtype=float)
    cand = np.asarray(cand_scores, dtype=float)
    if not (0 < pi_min <= pi_max <= 1):
        raise ValueError("need 0 < pi_min <= pi_max <= 1")
    k, n = cal.size, cand.size
    j = conformal_rank(k, alpha)
    t = np.sort(cal)[j - 1]
    K = int((cand > t).sum())
    r = certify_counts(k, j, K, n, alpha=alpha, tau=tau, pi_min=pi_min,
                       pi_max=pi_max, delta=delta, eta=eta, alpha_cap=alpha_cap,
                       sided=sided)
    code = int(r["decision"])
    return Certificate(float(t), k, j, n, K, float(r["fpr_lo"]), float(r["fpr_hi"]),
                       float(r["tpr_lo"]), float(r["tpr_hi"]),
                       float(r["flagged_poison_frac_lo"]), float(r["residual_poison_frac_hi"]),
                       {1: "GO", -1: "NO-GO", 0: "UNDETERMINED"}[code],
                       alpha, tau, pi_min, pi_max, delta, eta, sided)


def transport_check(trusted_member_scores, trusted_fresh_scores, t, delta=0.05):
    """Exact two-sided test of equal tail mass above t for members vs fresh trusted
    clean samples (Fisher exact).  Returns (gap_estimate, p_value).  A defender who
    has trusted members should calibrate on them; this check quantifies the gap."""
    a = np.asarray(trusted_member_scores); b = np.asarray(trusted_fresh_scores)
    x1, n1 = int((a > t).sum()), a.size
    x2, n2 = int((b > t).sum()), b.size
    p = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1]
    return x1 / n1 - x2 / n2, p


# ----------------------------------------------------------------------------
# realized-recall certificate (finite population, attack-agnostic in the poisons)
# ----------------------------------------------------------------------------
def certify_realized_counts(k, j, K, n, *, tau, n_poison_min, n_poison_max,
                            delta=0.05, alpha_cap=None):
    """Certificate for the REALIZED recall Kp/Np of a fixed (possibly adversarially
    planted) poison set.  Randomness comes only from the defender's secret, uniformly
    random choice of which clean training samples serve as trusted calibration.

    Exchangeability => the number M of non-calibration clean candidates above
    t = c_(j) is BetaBinomial(n_c, k-j+1, j), n_c = n - Np (exact, rank based, no
    distributional assumption, no assumption on the poison scores).  M is
    stochastically increasing in n_c, so bounds use the extreme n_c allowed by the
    declared poison-count range [n_poison_min, n_poison_max].
      flagged poisons Kp = K - M >= K - M_hi                (needs no ceiling)
      realized recall    >= (K - M_hi) / n_poison_max        (GO side)
      realized recall    <= (K - M_lo) / n_poison_min        (NO-GO side)
    Each one-sided statement fails with probability <= delta.  If alpha_cap is
    given, GO additionally requires the upper bound on the realized clean flag rate
    (from the same event M <= M_hi) to be <= alpha_cap.
    Requirements (see THEORY section 7.2): the trusted subset is a secret uniformly
    random subset of the clean training samples, accepted independently of score;
    the score map does not depend on which samples were selected; [n_poison_min,
    n_poison_max] refers to this fold's candidate set; alpha, tau, the detector and
    the range are fixed before looking at the calibration data.
    """
    a, b = k - j + 1, j
    nc_hi = n - n_poison_min
    nc_lo = n - n_poison_max
    m_hi = stats.betabinom.ppf(1 - delta, nc_hi, a, b)          # P(M > m_hi) <= delta
    m_lo = stats.betabinom.ppf(delta, nc_lo, a, b)              # P(M < m_lo) < delta
    kp_lo = np.maximum(np.asarray(K) - m_hi, 0)
    rec_lo = np.clip(kp_lo / max(n_poison_max, 1), 0, 1)
    rec_hi = np.clip((np.asarray(K) - m_lo) / max(n_poison_min, 1), 0, 1) if n_poison_min > 0 else np.ones_like(rec_lo)
    clean_fpr_hi = m_hi / max(nc_lo, 1)
    go = rec_lo >= tau
    if alpha_cap is not None:
        go = go & (clean_fpr_hi <= alpha_cap)
    nogo = rec_hi < tau
    decision = np.where(go, 1, np.where(nogo, -1, 0))
    return dict(flagged_poisons_lo=kp_lo, recall_lo=rec_lo, recall_hi=rec_hi,
                clean_flag_rate_hi=clean_fpr_hi, decision=decision)


def certify_realized(cal_scores, cand_scores, *, alpha=0.05, tau=0.8,
                     n_poison_min=0, n_poison_max, delta=0.05, alpha_cap=None, rng=None):
    """Raw-score wrapper; ties are broken uniformly at random (required for the
    exact rank law)."""
    rng = np.random.default_rng() if rng is None else rng
    cal = np.asarray(cal_scores, float); cand = np.asarray(cand_scores, float)
    k, n = cal.size, cand.size
    j = conformal_rank(k, alpha)
    # random tie-breaking by a tiny jitter on a common lexicographic key
    jc = rng.random(k); jx = rng.random(n)
    order = np.lexsort((jc, cal)); t_val, t_jit = cal[order[j - 1]], jc[order[j - 1]]
    above = (cand > t_val) | ((cand == t_val) & (jx > t_jit))
    K = int(above.sum())
    r = certify_realized_counts(k, j, K, n, tau=tau, n_poison_min=n_poison_min,
                                n_poison_max=n_poison_max, delta=delta, alpha_cap=alpha_cap)
    out = {kk: (float(v) if np.ndim(v) == 0 else v) for kk, v in r.items()}
    out.update(threshold=float(t_val), flagged=K, k=k, j=j, n=n,
               decision_label={1: "GO", -1: "NO-GO", 0: "UNDETERMINED"}[int(r["decision"])])
    return out


# ----------------------------------------------------------------------------
# theory helpers
# ----------------------------------------------------------------------------
def _kl_bern(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def required_trusted_size(alpha, pi, margin, delta=0.05, power=0.8, split=0.5,
                          roc_slope=0.0):
    """First-order trusted size for the one-sided rank-threshold certificate
    (n -> infinity, pi known): k = alpha(1-alpha)[c z1 + (c + rho) z_beta]^2 / m^2,
    c = (1-pi)/pi, rho = ROC slope dTPR/dFPR at alpha, z1 = z_{1-delta*split}."""
    z1 = stats.norm.ppf(1 - delta * split)
    zb = stats.norm.ppf(power)
    c = (1 - pi) / pi
    return alpha * (1 - alpha) * (c * z1 + (c + roc_slope) * zb) ** 2 / margin ** 2


def lecam_interval_gap(alpha, pi, k, delta=0.05, gamma=0.5):
    """Theorem 4 (interval form): largest TPR gap g = (1-pi)h/pi such that any
    interval with coverage >= 1-delta in both worlds has |I| >= g with probability
    > 1 - gamma; requires sqrt(k*kl(alpha||alpha+h)/2) < gamma - 2*delta."""
    target = gamma - 2 * delta
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 1 - alpha - 1e-12
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.sqrt(k * _kl_bern(alpha, alpha + mid) / 2) < target:
            lo = mid
        else:
            hi = mid
    return (1 - pi) * lo / pi


def lecam_k_lower_bound_test(alpha, pi, margin, delta=0.05, power=0.8):
    """Theorem 4 (test form): any GO rule with P(false GO) <= delta in the world with
    TPR' = TPR - margin and P(GO) >= power in the world with TPR needs
    TV >= power - delta, hence k >= 2(power-delta)^2 / kl(alpha || alpha + h),
    h = pi*margin/(1-pi).  Valid only where the tail-density condition
    f1/f0 >= margin/alpha holds on {s > t}."""
    h = pi * margin / (1 - pi)
    if alpha + h >= 1:
        return float("nan")
    return 2 * (power - delta) ** 2 / _kl_bern(alpha, alpha + h)
