"""Mock S1 demonstration — SYNTHETIC scores only (no models, no backdoors).

Purpose: show exactly what a real S1 run produces, and what the one hand-off file
looks like, so a first-time user can see the finished product.  The ONLY thing
that changes in a real run is that `score` comes from your own detector on your own
victim models instead of from this synthetic generator.  Every other column, the
command, and the output format are identical.
"""
import sys, pathlib, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.s1_analysis_template import run, RATIOS

OUT = pathlib.Path(os.environ.get("P14CERT_OUT", pathlib.Path(__file__).resolve().parents[1] / "results"))
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
z = stats.norm.isf(ALPHA)

# Each scenario stands in for one "dataset x detector" cell.  target_recall is the
# detector's true recall at FPR = alpha; member_shift is how much higher trained
# samples score than fresh ones (the membership effect the certificate must survive).
SCENARIOS = [
    dict(name="strong_detector",  pi=0.10, target_recall=0.95, member_shift=0.0,  note="好检测器：应给 GO"),
    dict(name="weak_detector",    pi=0.10, target_recall=0.40, member_shift=0.0,  note="差检测器：应给 NO-GO"),
    dict(name="borderline",       pi=0.05, target_recall=0.82, member_shift=0.0,  note="临界：k 小时应弃权"),
    dict(name="membership_trap",  pi=0.05, target_recall=0.70, member_shift=0.15, note="成员偏移陷阱：新样本校准会虚假 GO"),
]
N_CLEAN_TRAIN = int(os.environ.get("MOCK_NC", 20000))
N_FRESH = int(os.environ.get("MOCK_NF", 20000))
SEEDS = int(os.environ.get("MOCK_SEEDS", 3))


def make_scores(sc, seed):
    rng = np.random.default_rng([2026, seed, hash(sc["name"]) % 9973])
    n_p = int(round(sc["pi"] * N_CLEAN_TRAIN / (1 - sc["pi"])))
    mu_p = z + stats.norm.isf(1 - sc["target_recall"])   # poison mean to hit target recall
    rows = []
    # trained clean samples: shifted up by member_shift
    s = rng.normal(sc["member_shift"], 1, N_CLEAN_TRAIN)
    rows += [("train", float(v), 0) for v in s]
    # poisoned training samples
    s = rng.normal(mu_p, 1, n_p)
    rows += [("train", float(v), 1) for v in s]
    # fresh clean held-out samples (never trained on)
    s = rng.normal(0.0, 1, N_FRESH)
    rows += [("fresh", float(v), 0) for v in s]
    df = pd.DataFrame(rows, columns=["role", "score", "is_poison"])
    df.insert(0, "sample_id", np.arange(len(df)))
    df.insert(0, "fold", 0)
    df.insert(0, "victim_id", f"{sc['name']}_seed{seed}")
    return df


if __name__ == "__main__":
    allscores = pd.concat([make_scores(sc, s) for sc in SCENARIOS for s in range(SEEDS)], ignore_index=True)
    scores_path = OUT / "mock_s1_scores.csv"
    allscores.to_csv(scores_path, index=False)
    print(f"wrote synthetic hand-off file: {scores_path}  ({len(allscores):,} rows)")

    # this is the exact command a real run uses
    res = run(str(scores_path), str(OUT / "mock_s1_report_raw.csv"), ks=[120, 2000], R=int(os.environ.get("MOCK_R", 200)))
    rep = pd.DataFrame(res)

    # summarise to one row per scenario (pool seeds)
    rep["scenario"] = rep.victim_id.str.replace(r"_seed\d+", "", regex=True)
    out = []
    for (scn, k), g in rep.groupby(["scenario", "k"]):
        note = next(s["note"] for s in SCENARIOS if s["name"] == scn)
        out.append(dict(
            scenario=scn, note=note, k=k,
            true_recall=round(g["mean_realized_recall"].mean(), 3),
            member_minus_fresh_fpr=round(g["mean_member_minus_fresh"].mean(), 4),
            go_rate=round(g["dec_r1.2_go_rate"].mean(), 3),
            nogo_rate=round(g["dec_r1.2_nogo_rate"].mean(), 3),
            false_go_rate=round(g["false_go_r1.2_rate"].mean(), 4),
            caught_lower_bound=round(g.get("mean_caught_lo_r1.0", pd.Series([np.nan])).mean(), 1),
            fresh_calib_false_go=round(g.get("fresh_false_go_r1.2_rate", pd.Series([np.nan])).mean(), 3),
        ))
    summary = pd.DataFrame(out).sort_values(["scenario", "k"])
    summary.to_csv(OUT / "mock_s1_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== MOCK S1 SUMMARY (synthetic) ===")
    print(summary.to_string(index=False))
