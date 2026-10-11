"""S1-S4 analysis template (to be run by the paper14 team on their own score tables).

Input CSV columns (one row per scored sample, per victim and fold):
  victim_id, fold, sample_id, score, role, is_poison
    role      : "train" (sample in the training set) or "fresh" (clean, not trained on)
    is_poison : 0/1 ground truth, used ONLY for evaluation, never by the certificate
Higher score = more suspicious.  For each victim/fold the script repeatedly draws a
secret uniformly random trusted subset of k clean training samples (this simulates
the defender's random verified sample), certifies the remaining training samples,
and records realized recall, decisions and the member-vs-fresh FPR gap.

Usage:  python s1_analysis_template.py scores.csv out.csv --k 120 500 2000 --R 200
Self-test on synthetic data:  python s1_analysis_template.py --selftest
"""
import argparse, sys, pathlib, csv
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cert.certificate import conformal_rank, certify_realized_counts

ALPHA, TAU, DELTA, CAP = 0.05, 0.8, 0.05, 0.10
RATIOS = (1.0, 1.2, 2.0, 4.0)        # declared N_max / N_p ; N_min = N_p / 2


def _ranks_with_random_ties(scores, rng):
    """Replace scores by their ranks under one global random tie-breaking order."""
    order = np.lexsort((rng.random(scores.size), scores))
    r = np.empty(scores.size); r[order] = np.arange(scores.size)
    return r


def _certify_from_cal(s, cal_idx, cand_mask, is_poison, n_p, k, j, d, row, prefix,
                      fpr_mode="loose"):
    t = np.sort(s[cal_idx])[j - 1]
    flagged = cand_mask & (s > t)
    K = int(flagged.sum()); n = int(cand_mask.sum())
    kp = int((flagged & (is_poison == 1)).sum())
    rec = kp / n_p
    clean_flag = float((flagged & (is_poison == 0)).sum() / max((cand_mask & (is_poison == 0)).sum(), 1))
    row[prefix + "realized_recall"] = rec
    for ratio in RATIOS:
        c = certify_realized_counts(k, j, K, n, tau=TAU, n_poison_min=n_p // 2,
                                    n_poison_max=int(np.ceil(ratio * n_p)), delta=d,
                                    alpha_cap=CAP, fpr_mode=fpr_mode)
        dec = int(c["decision"])
        row[f"{prefix}dec_r{ratio}"] = dec
        row[f"{prefix}false_go_r{ratio}"] = int(dec == 1 and (rec < TAU or clean_flag > CAP))
        row[f"{prefix}caught_lo_r{ratio}"] = float(c["flagged_poisons_lo"])
    return t


def audit_group(scores, role, is_poison, k, R, rng, folds_total=1,
                fpr_mode="loose"):
    """Evaluation only: trusted subsets are drawn from KNOWN-clean training samples,
    so exchangeability holds by construction (this checks implementation, not the
    selection premise; see PREREG S3 item 3).  Two calibration sources share the
    same certificate (Prop 7 counting): (i) trusted members, (ii) fresh clean
    samples (violates exchangeability if members score differently)."""
    out = []
    train = role == "train"
    clean_tr = np.where(train & (is_poison == 0))[0]
    pois = np.where(train & (is_poison == 1))[0]
    fresh = np.where(role == "fresh")[0]
    n_p = len(pois)
    if len(clean_tr) <= k or n_p == 0:
        return out
    if fresh.size and fresh.size < 5 * k:
        print(f"warning: fresh pool {fresh.size} < 5k={5*k}; fresh-calibration draws overlap heavily",
              file=sys.stderr)
    j = conformal_rank(k, ALPHA)
    d = DELTA / folds_total
    s = _ranks_with_random_ties(np.asarray(scores, float), rng)
    for _ in range(R):
        tr = rng.choice(clean_tr, k, replace=False)
        cand_mask = train.copy(); cand_mask[tr] = False
        row = dict(k=k, caught_true=None)
        t = _certify_from_cal(s, tr, cand_mask, is_poison, n_p, k, j, d, row, "",
                              fpr_mode=fpr_mode)
        row["caught_true"] = int((cand_mask & (s > t) & (is_poison == 1)).sum())
        fpr_member = float((s[np.setdiff1d(clean_tr, tr)] > t).mean())
        fpr_fresh = float((s[fresh] > t).mean()) if fresh.size else np.nan
        row.update(fpr_member=fpr_member, fpr_fresh=fpr_fresh, member_minus_fresh=fpr_member - fpr_fresh)
        if fresh.size > k:
            fc = rng.choice(fresh, k, replace=False)
            _certify_from_cal(s, fc, train.copy(), is_poison, n_p, k, j, d, row, "fresh_",
                              fpr_mode=fpr_mode)
        out.append(row)
    return out


def summarise(rows):
    """Per victim/fold/k: false-GO counts out of R (exact binomial test vs delta/F is
    applied downstream with Holm correction across cells), GO rates, means."""
    summ = {"R": len(rows)}
    for kk in rows[0]:
        if "false_go" in kk:
            v = np.array([r[kk] for r in rows if kk in r], float)
            summ[kk + "_count"] = int(v.sum()); summ[kk + "_rate"] = float(v.mean())
        elif "dec_" in kk:
            v = np.array([r[kk] for r in rows if kk in r], float)
            summ[kk + "_go_rate"] = float(np.mean(v == 1)); summ[kk + "_nogo_rate"] = float(np.mean(v == -1))
    for kk in ("realized_recall", "fresh_realized_recall", "member_minus_fresh", "caught_true",
               "caught_lo_r1.0", "caught_lo_r2.0"):
        vals = [
            r[kk]
            for r in rows
            if kk in r and r[kk] is not None and np.isfinite(float(r[kk]))
        ]
        if vals:
            summ["mean_" + kk] = float(np.nanmean(vals))
    return summ


def run(path_in, path_out, ks, R, seed=0, fpr_mode="loose"):
    import pandas as pd
    df = pd.read_csv(path_in)
    rng = np.random.default_rng(seed)
    res = []
    F = df.groupby("victim_id")["fold"].nunique().max()
    for (vid, fold), g in df.groupby(["victim_id", "fold"]):
        for k in ks:
            rows = audit_group(g.score.to_numpy(float), g.role.to_numpy(), g.is_poison.to_numpy(int),
                               k, R, rng, folds_total=F, fpr_mode=fpr_mode)
            if rows:
                res.append(dict(victim_id=vid, fold=fold, k=k, **summarise(rows)))
    with open(path_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    return res


def selftest():
    """Synthetic score table: 3 victims x 2 folds.  Member clean scores are shifted by
    +0.2 relative to fresh clean scores; 79% of poisons are far above, 21% far below,
    so realized recall is pinned at 0.79 < tau = 0.8 (a boundary case)."""
    import pandas as pd, tempfile, os
    rng = np.random.default_rng(1)
    rows = []
    for v in range(3):
        for f in range(2):
            nc, npz, nf = 20000, 2000, 30000
            sc = np.r_[rng.normal(0.2, 1, nc), np.full(int(0.79 * npz), 8.0),
                       np.full(npz - int(0.79 * npz), -8.0), rng.normal(0, 1, nf)]
            role = ["train"] * (nc + npz) + ["fresh"] * nf
            lab = [0] * nc + [1] * npz + [0] * nf
            rows += [dict(victim_id=v, fold=f, sample_id=i, score=sc[i], role=role[i], is_poison=lab[i])
                     for i in range(sc.size)]
    tmp = tempfile.mkdtemp()
    pin, pout = os.path.join(tmp, "s.csv"), os.path.join(tmp, "o.csv")
    pd.DataFrame(rows).to_csv(pin, index=False)
    res = run(pin, pout, ks=[2000, 5000], R=200)
    r = pd.DataFrame(res)
    cols = ["mean_realized_recall", "mean_member_minus_fresh", "false_go_r1.0_rate",
            "fresh_false_go_r1.0_rate", "dec_r1.0_go_rate"]
    print(r.groupby("k")[cols].mean().round(3))
    member_fg = r["false_go_r1.0_count"].sum() / r["R"].sum()
    fresh_fg = r["fresh_false_go_r1.0_count"].sum() / r["R"].sum()
    print(f"pooled false-GO: trusted-member calibration {member_fg:.3f}, fresh calibration {fresh_fg:.3f}")
    assert member_fg <= DELTA / 2 + 0.02, "member calibration should be valid at delta/F"
    assert fresh_fg > DELTA, "fresh calibration should fail under a +0.2 member shift"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--k", type=int, nargs="+", default=[120, 500, 2000])
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--fpr-mode", choices=("loose", "tight"), default="loose")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.inp, a.out, a.k, a.R, fpr_mode=a.fpr_mode)
