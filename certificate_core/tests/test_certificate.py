import math
import sys, pathlib
import numpy as np
from scipy import stats
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import (conformal_rank, fpr_beta_bounds, clopper_pearson,
                              certify_counts, certify, required_trusted_size,
                              lecam_interval_gap, lecam_k_lower_bound_test,
                              transport_check, certify_realized_counts, certify_realized)

RNG = np.random.default_rng(20261007)


def test_conformal_rank_and_minimum_k():
    assert conformal_rank(120, 0.05) == 115
    assert conformal_rank(19, 0.05) == 19
    with pytest.raises(ValueError):
        conformal_rank(18, 0.05)


def test_beta_law_of_fpr_at_conformal_threshold():
    # FPR(c_(j)) for Uniform(0,1) clean scores is 1 - U_(j) ~ Beta(k-j+1, j)
    k, alpha = 120, 0.05
    j = conformal_rank(k, alpha)
    u = np.sort(RNG.random((20000, k)), axis=1)[:, j - 1]
    fpr = 1 - u
    p = stats.kstest(fpr, stats.beta(k - j + 1, j).cdf).pvalue
    assert p > 0.001
    assert fpr.mean() <= alpha + 3 * fpr.std() / math.sqrt(fpr.size)  # conformal validity


def test_clopper_pearson_coverage():
    n, p, d = 500, 0.07, 0.05
    K = RNG.binomial(n, p, size=20000)
    lo, hi = clopper_pearson(K, n, d)
    cov = np.mean((lo <= p) & (p <= hi))
    assert cov >= 1 - d - 0.005


def test_bounds_ordered_and_tighten():
    r = certify_counts(10**6, conformal_rank(10**6, 0.05), 140_000, 10**6,
                       alpha=0.05, tau=0.8, pi_min=0.1, pi_max=0.1)
    # true: q = 0.9*0.05 + 0.1*TPR = 0.14 -> TPR = 0.95
    assert r["tpr_lo"] <= r["tpr_hi"]
    assert abs(float(r["tpr_lo"]) - 0.95) < 0.02 and abs(float(r["tpr_hi"]) - 0.95) < 0.02
    assert int(r["decision"]) == 1


def test_raw_equals_counts():
    cal = RNG.normal(size=200)
    cand = np.r_[RNG.normal(size=900), RNG.normal(3, 1, size=100)]
    c = certify(cal, cand, alpha=0.05, tau=0.8, pi_min=0.05, pi_max=0.2)
    j = conformal_rank(200, 0.05)
    t = np.sort(cal)[j - 1]
    r = certify_counts(200, j, int((cand > t).sum()), cand.size, alpha=0.05, tau=0.8,
                       pi_min=0.05, pi_max=0.2)
    assert math.isclose(c.tpr_lo, float(r["tpr_lo"])) and c.flagged == int((cand > t).sum())


def test_interval_coverage_gaussian_raw():
    # end-to-end raw-sample coverage of [tpr_lo, tpr_hi] for population TPR(t)
    k, n, pi, mu, alpha, delta = 300, 4000, 0.1, 2.5, 0.05, 0.05
    hits = 0; R = 1500
    for _ in range(R):
        cal = RNG.normal(size=k)
        npz = RNG.binomial(n, pi)
        cand = np.r_[RNG.normal(size=n - npz), RNG.normal(mu, 1, size=npz)]
        c = certify(cal, cand, alpha=alpha, tau=0.8, pi_min=pi, pi_max=pi, delta=delta,
                    sided="two")
        tpr_true = stats.norm.sf(c.threshold - mu)
        hits += (c.tpr_lo <= tpr_true <= c.tpr_hi)
    assert hits / R >= 1 - delta - 0.015


def test_one_sided_lower_bound_validity():
    k, n, pi, mu, alpha, delta = 200, 3000, 0.1, 2.0, 0.05, 0.05
    j = conformal_rank(k, alpha)
    u = RNG.beta(k - j + 1, j, size=40000)
    t = stats.norm.isf(u); tpr = stats.norm.sf(t - mu)
    K = RNG.binomial(n, (1 - pi) * u + pi * tpr)
    r = certify_counts(k, j, K, n, alpha=alpha, tau=0.8, pi_min=pi, pi_max=pi, delta=delta)
    assert np.mean(r["tpr_lo"] > tpr) <= delta + 0.005
    assert np.mean(r["tpr_hi"] < tpr) <= delta + 0.005


def test_transport_bias_breaks_and_eta_repairs():
    # clean candidates shifted up by s: fresh calibration without eta over-states TPR
    k, n, pi, mu, s, alpha = 5000, 200000, 0.05, 2.0, 0.15, 0.05
    j = conformal_rank(k, alpha)
    u = RNG.beta(k - j + 1, j, size=4000)
    t = stats.norm.isf(u)
    fpr_c = stats.norm.sf(t - s)                 # clean-candidate FPR
    tpr = stats.norm.sf(t - mu)
    K = RNG.binomial(n, (1 - pi) * fpr_c + pi * tpr)
    r0 = certify_counts(k, j, K, n, alpha=alpha, tau=0.5, pi_min=pi, pi_max=pi, eta=0.0)
    gap = float(np.max(np.abs(fpr_c - u)))
    r1 = certify_counts(k, j, K, n, alpha=alpha, tau=0.5, pi_min=pi, pi_max=pi, eta=gap)
    cov0 = np.mean((r0["tpr_lo"] <= tpr) & (tpr <= r0["tpr_hi"]))
    cov1 = np.mean((r1["tpr_lo"] <= tpr) & (tpr <= r1["tpr_hi"]))
    assert cov0 < 0.5 and cov1 >= 0.95


def test_theory_helpers_monotone():
    a = required_trusted_size(0.05, 0.05, 0.05)
    b = required_trusted_size(0.05, 0.10, 0.05)
    assert a > 3.5 * b                      # ~ (1-pi)^2/pi^2 scaling
    w1 = lecam_interval_gap(0.05, 0.05, 100)
    w2 = lecam_interval_gap(0.05, 0.05, 400)
    assert 1.7 < w1 / w2 < 2.3              # ~ 1/sqrt(k)
    k1 = lecam_k_lower_bound_test(0.05, 0.05, 0.05)
    k2 = lecam_k_lower_bound_test(0.05, 0.10, 0.05)
    assert 3.0 < k1 / k2 < 4.5


def test_transport_check_runs():
    g, p = transport_check(RNG.normal(0.2, 1, 3000), RNG.normal(0, 1, 3000), 1.645)
    assert g > 0 and p < 0.05


def test_betabinomial_law_finite_population():
    # fixed population of clean scores; random secret trusted subset; the count of
    # remaining clean above c_(j) must follow BetaBinomial(N-k, k-j+1, j) exactly
    N, k, alpha = 400, 60, 0.05
    j = conformal_rank(k, alpha)
    pop = RNG.standard_t(3, size=N)            # any fixed values (no ties)
    counts = []
    for _ in range(20000):
        idx = RNG.permutation(N)
        cal, rest = pop[idx[:k]], pop[idx[k:]]
        t = np.sort(cal)[j - 1]
        counts.append(int((rest > t).sum()))
    counts = np.asarray(counts)
    bb = stats.betabinom(N - k, k - j + 1, j)
    for m in (0, 5, 15, 30):
        assert abs(np.mean(counts <= m) - bb.cdf(m)) < 0.012


def test_realized_certificate_adversarial_planted_set():
    # adversary fixes poison scores so that realized recall is 0.79 (< tau = 0.8):
    # 79% of poisons far above any clean score, 21% far below; GO must be rare.
    k, alpha, delta, tau = 200, 0.05, 0.05, 0.8
    n_clean_total, n_p = 20000, 1000
    bad = 0; R = 2000
    for _ in range(R):
        clean = RNG.normal(size=n_clean_total)
        idx = RNG.permutation(n_clean_total)
        cal, clean_cand = clean[idx[:k]], clean[idx[k:]]
        poisons = np.r_[np.full(790, 50.0), np.full(210, -50.0)]
        cand = np.r_[clean_cand, poisons]
        r = certify_realized(cal, cand, alpha=alpha, tau=tau, n_poison_min=500,
                             n_poison_max=n_p, delta=delta, rng=RNG)
        bad += (r["decision_label"] == "GO")
    assert bad / R <= delta + 0.01


def test_tight_fpr_is_valid_and_tighter_than_loose():
    # realized clean-flag-rate upper bound: tight <= loose, and tight still covers
    from cert.certificate import _realized_clean_fpr_upper
    k, alpha, delta = 120, 0.05, 0.025
    j = conformal_rank(k, alpha)
    a, b = k - j + 1, j
    n = 1465; n_min, n_max = 150, 300          # candidate clean count in [n-nmax, n-nmin]
    nc_lo, nc_hi = n - n_max, n - n_min
    tight, m_hi_t = _realized_clean_fpr_upper(a, b, nc_lo, nc_hi, delta, "tight")
    loose, m_hi_l = _realized_clean_fpr_upper(a, b, nc_lo, nc_hi, delta, "loose")
    assert tight <= loose and m_hi_t == m_hi_l          # count bound unchanged
    assert loose > 0.10 and tight < loose               # loose blocks alpha_cap=0.10, tight helps
    # coverage: for every true clean count n_c in range, P(rate > tight) <= delta
    for n_c in (nc_lo, (nc_lo + nc_hi) // 2, nc_hi):
        M = stats.betabinom(n_c, a, b).rvs(size=30000, random_state=RNG)
        assert np.mean(M / n_c > tight) <= delta + 0.004


def test_fast_betabinomial_ppf_matches_scipy():
    from cert.certificate import _betabinom_ppf_many
    for a, b in ((1, 19), (6, 115), (24, 477), (100, 1901)):
        ns = np.asarray([1, 7, 31, 120, 500, 2000])
        expected = stats.betabinom.ppf(0.975, ns, a, b).astype(int)
        actual = _betabinom_ppf_many(0.975, ns, a, b)
        assert np.array_equal(actual, expected)


def test_tight_fpr_unblocks_go_without_false_go():
    # strong detector, k where recall clears tau but loose FPR cap blocks GO
    k, alpha, delta, tau = 120, 0.05, 0.025, 0.8
    n_clean_total, n_p = 2330, 300
    j = conformal_rank(k, alpha)
    bad_loose = bad_tight = go_loose = go_tight = 0; R = 1500
    for _ in range(R):
        clean = RNG.normal(size=n_clean_total)
        pois = RNG.normal(5.0, 1.0, size=n_p)              # well separated: recall ~ 1
        idx = RNG.permutation(n_clean_total)
        cal, clean_cand = clean[idx[:k]], clean[idx[k:]]
        cand = np.r_[clean_cand, pois]
        rec = (pois > np.sort(cal)[j - 1]).mean()
        for mode, acc in (("loose", "loose"), ("tight", "tight")):
            r = certify_realized(cal, cand, alpha=alpha, tau=tau, n_poison_min=n_p // 2,
                                 n_poison_max=n_p, delta=delta, alpha_cap=0.10,
                                 fpr_mode=mode, rng=RNG)
            if mode == "loose":
                go_loose += r["decision_label"] == "GO"; bad_loose += (r["decision_label"] == "GO" and rec < tau)
            else:
                go_tight += r["decision_label"] == "GO"; bad_tight += (r["decision_label"] == "GO" and rec < tau)
    assert go_tight > go_loose                     # tight unblocks GO
    assert bad_tight / R <= delta + 0.01           # still no false GO
