"""Matched comparison with Liu et al. (JMLR 2022), Algorithm 1.

Both methods receive the same i.i.d. clean and mixture score tables.  The common
operating target is population recall >= TAU and population FPR <= FPR_CAP.

* ``ours`` fixes the split-conformal FPR operating point, then applies the
  population certificate in ``cert/certificate.py``.
* ``liu`` implements the paper's Basic-CDF Algorithm 1 with the Corollary 1
  finite-sample correction.  It selects q = (1-TAU)-epsilon and emits GO only
  when the corollary's induced FPR upper bound is <= FPR_CAP.  The original
  method has no NO-GO output, so every other outcome is ABSTAIN.

The comparison is deliberately restricted to Liu et al.'s balanced setting
|S0|=|Sm|=n and exact prevalence.  It does not claim that the two procedures
choose the same threshold or certify the same random object.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import time

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from cert.certificate import certify_counts, conformal_rank


ALPHA = 0.05
FPR_CAP = 0.10
TAU = 0.80
DELTA = 0.05

FAMILIES = ("binormal", "binormal_unequal", "partial_40pct", "t3")
PIS = (0.05, 0.10, 0.20)
MUS = (2.0, 4.0)
SIZES = (2_000, 10_000)


def liu_corrections(n: int, prevalence: float, delta: float = DELTA) -> tuple[float, float]:
    """Recall and FPR corrections from Liu et al., Corollary 1.

    The displayed corollary splits delta between the alien-recall guarantee and
    the induced-FPR estimate.  Its recall correction is obtained by solving the
    sample-size inequality for epsilon; epsilon_0 is the DKW FPR radius.
    """
    log_term = math.log(2.0 / (1.0 - math.sqrt(1.0 - delta / 2.0)))
    eps_recall = ((2.0 - prevalence) / prevalence) * math.sqrt(log_term / (2.0 * n))
    eps_fpr = math.sqrt(math.log(4.0 / delta) / (2.0 * n))
    return eps_recall, eps_fpr


def liu_minimum_balanced_n(prevalence: float, tau: float = TAU, delta: float = DELTA) -> int:
    """Smallest integer n for which Corollary 1 allows a positive q."""
    eta = 1.0 - tau
    log_term = math.log(2.0 / (1.0 - math.sqrt(1.0 - delta / 2.0)))
    boundary = 0.5 * log_term * (((2.0 - prevalence) / prevalence) / eta) ** 2
    return math.floor(boundary) + 1


def sample_clean(rng: np.random.Generator, family: str, size: int) -> np.ndarray:
    if family == "t3":
        return rng.standard_t(3, size=size)
    return rng.normal(size=size)


def sample_poison(rng: np.random.Generator, family: str, mu: float, size: int) -> np.ndarray:
    if family == "binormal":
        return rng.normal(mu, 1.0, size=size)
    if family == "binormal_unequal":
        return rng.normal(mu, 2.0, size=size)
    if family == "partial_40pct":
        shifted = rng.random(size) < 0.6
        out = rng.normal(size=size)
        out[shifted] += mu
        return out
    if family == "t3":
        return rng.standard_t(3, size=size) + mu
    raise ValueError(f"unknown family: {family}")


def population_fpr(family: str, threshold: float) -> float:
    if threshold == -math.inf:
        return 1.0
    if family == "t3":
        return float(stats.t(3).sf(threshold))
    return float(stats.norm.sf(threshold))


def population_tpr(family: str, mu: float, threshold: float) -> float:
    if threshold == -math.inf:
        return 1.0
    if family == "binormal":
        return float(stats.norm(mu, 1.0).sf(threshold))
    if family == "binormal_unequal":
        return float(stats.norm(mu, 2.0).sf(threshold))
    if family == "partial_40pct":
        return float(0.6 * stats.norm(mu, 1.0).sf(threshold) + 0.4 * stats.norm.sf(threshold))
    if family == "t3":
        return float(stats.t(3, loc=mu).sf(threshold))
    raise ValueError(f"unknown family: {family}")


def liu_algorithm1(clean: np.ndarray, mixture: np.ndarray, prevalence: float, q: float) -> tuple[float, float]:
    """Basic-CDF Algorithm 1; returns threshold and empirical clean CDF there."""
    n0, nm = clean.size, mixture.size
    values = np.concatenate((clean, mixture))
    is_mix = np.concatenate((np.zeros(n0, dtype=np.int8), np.ones(nm, dtype=np.int8)))
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_mix = is_mix[order]
    cum_mix = np.cumsum(ordered_mix)
    cum_clean = np.arange(1, n0 + nm + 1) - cum_mix

    # Empirical CDFs include every observation tied at u.  The generators are
    # continuous, but grouping makes the implementation exact for accidental ties.
    group_end = np.r_[ordered_values[1:] != ordered_values[:-1], True]
    idx = np.flatnonzero(group_end)
    f_mix = cum_mix[idx] / nm
    f_clean = cum_clean[idx] / n0
    f_alien = (f_mix - (1.0 - prevalence) * f_clean) / prevalence
    admissible = idx[f_alien <= q]
    if admissible.size == 0:
        return -math.inf, 0.0
    chosen = int(admissible[-1])
    return float(ordered_values[chosen]), float(cum_clean[chosen] / n0)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cell(family: str, prevalence: float, mu: float, n: int, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    j = conformal_rank(n, ALPHA)
    eps_recall, eps_fpr = liu_corrections(n, prevalence)
    q = 1.0 - TAU - eps_recall
    liu_feasible = q > 0.0

    ours_go = ours_false = ours_nogo = 0
    liu_go = liu_false = 0
    ours_width = []
    ours_fpr_width = []
    liu_fpr_upper = []
    liu_tpr = []
    liu_fpr = []

    nominal_threshold = stats.t(3).isf(ALPHA) if family == "t3" else stats.norm.isf(ALPHA)
    nominal_tpr = population_tpr(family, mu, float(nominal_threshold))

    for _ in range(replicates):
        clean = sample_clean(rng, family, n)
        mask = rng.random(n) < prevalence
        mixture = sample_clean(rng, family, n)
        if mask.any():
            mixture[mask] = sample_poison(rng, family, mu, int(mask.sum()))

        threshold_ours = float(np.partition(clean, j - 1)[j - 1])
        flagged = int(np.count_nonzero(mixture > threshold_ours))
        cert = certify_counts(
            n,
            j,
            flagged,
            n,
            alpha=ALPHA,
            tau=TAU,
            pi_min=prevalence,
            pi_max=prevalence,
            delta=DELTA,
            alpha_cap=FPR_CAP,
        )
        decision = int(cert["decision"])
        actual_fpr = population_fpr(family, threshold_ours)
        actual_tpr = population_tpr(family, mu, threshold_ours)
        bad = actual_fpr > FPR_CAP or actual_tpr < TAU
        ours_go += decision == 1
        ours_nogo += decision == -1
        ours_false += decision == 1 and bad
        ours_width.append(float(cert["tpr_hi"] - cert["tpr_lo"]))
        ours_fpr_width.append(float(cert["fpr_hi"] - cert["fpr_lo"]))

        if liu_feasible:
            threshold_liu, fhat0 = liu_algorithm1(clean, mixture, prevalence, q)
            fpr_hi = min(1.0, 1.0 - fhat0 + eps_fpr)
            actual_fpr_liu = population_fpr(family, threshold_liu)
            actual_tpr_liu = population_tpr(family, mu, threshold_liu)
            go = fpr_hi <= FPR_CAP
            bad_liu = actual_fpr_liu > FPR_CAP or actual_tpr_liu < TAU
            liu_go += go
            liu_false += go and bad_liu
            liu_fpr_upper.append(fpr_hi)
            liu_tpr.append(actual_tpr_liu)
            liu_fpr.append(actual_fpr_liu)

    return {
        "family": family,
        "pi": prevalence,
        "mu": mu,
        "n_clean": n,
        "n_mixture": n,
        "replicates": replicates,
        "nominal_tpr_at_fpr_0.05": nominal_tpr,
        "nominal_strong": nominal_tpr >= TAU + 0.05,
        "ours_go_rate": ours_go / replicates,
        "ours_nogo_rate": ours_nogo / replicates,
        "ours_false_assurance_rate": ours_false / replicates,
        "ours_mean_tpr_interval_width": float(np.mean(ours_width)),
        "ours_mean_fpr_interval_width": float(np.mean(ours_fpr_width)),
        "liu_cor1_feasible": liu_feasible,
        "liu_q": q,
        "liu_recall_correction": eps_recall,
        "liu_fpr_radius": eps_fpr,
        "liu_go_rate": liu_go / replicates,
        "liu_false_assurance_rate": liu_false / replicates,
        "liu_mean_population_tpr": float(np.mean(liu_tpr)) if liu_tpr else math.nan,
        "liu_mean_population_fpr": float(np.mean(liu_fpr)) if liu_fpr else math.nan,
        "liu_mean_fpr_upper": float(np.mean(liu_fpr_upper)) if liu_fpr_upper else math.nan,
    }


def summarize(cells: list[dict]) -> list[dict]:
    rows = []
    for n in SIZES:
        for prevalence in PIS:
            group = [r for r in cells if r["n_clean"] == n and r["pi"] == prevalence and r["nominal_strong"]]
            total = sum(int(r["replicates"]) for r in group)
            if not total:
                continue
            weighted = lambda key: sum(float(r[key]) * int(r["replicates"]) for r in group) / total
            rows.append({
                "pi": prevalence,
                "n_clean=n_mixture": n,
                "strong_cells": len(group),
                "total_replicates": total,
                "ours_go_rate": weighted("ours_go_rate"),
                "ours_false_assurance_rate": weighted("ours_false_assurance_rate"),
                "ours_mean_tpr_interval_width": weighted("ours_mean_tpr_interval_width"),
                "liu_cor1_feasible": all(bool(r["liu_cor1_feasible"]) for r in group),
                "liu_go_rate": weighted("liu_go_rate"),
                "liu_false_assurance_rate": weighted("liu_false_assurance_rate"),
                "liu_recall_correction": weighted("liu_recall_correction"),
                "liu_fpr_radius": weighted("liu_fpr_radius"),
            })
    return rows


def budget_rows() -> list[dict]:
    source = ROOT / "results" / "s0b_kstar.csv"
    ours = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["margin"]) - 0.05) < 1e-12:
                ours[float(row["pi"])] = float(row["k_star_empirical"])
    rows = []
    for prevalence in (0.01, 0.02, 0.05, 0.10, 0.20):
        rows.append({
            "pi": prevalence,
            "ours_empirical_kstar_80pct_power_margin0.05": ours[prevalence],
            "liu_cor1_min_balanced_n_for_positive_q": liu_minimum_balanced_n(prevalence),
            "liu_total_clean_plus_mixture": 2 * liu_minimum_balanced_n(prevalence),
        })
    return rows


def render_report(path: pathlib.Path, summary: list[dict], budgets: list[dict], replicates: int) -> None:
    lines = [
        "# Liu Algorithm 1 matched baseline",
        "",
        "This run implements the Basic-CDF Algorithm 1 and Corollary 1 from Liu et al. (JMLR 2022).",
        "Both procedures receive the same clean and mixture score tables. Exact prevalence is supplied to both.",
        f"Each cell uses {replicates} replications. The common operating target is recall >= {TAU:.2f} and FPR <= {FPR_CAP:.2f}.",
        "Only cells whose nominal 5% FPR threshold has recall >= 0.85 enter the power summary.",
        "",
        "## Matched joint decision",
        "",
        "| pi | n clean = n mixture | strong cells | ours GO | Liu GO | ours false assurance | Liu false assurance | Liu Cor. 1 feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['pi']:.2f} | {row['n_clean=n_mixture']:,} | {row['strong_cells']} | "
            f"{100*row['ours_go_rate']:.2f}% | {100*row['liu_go_rate']:.2f}% | "
            f"{100*row['ours_false_assurance_rate']:.2f}% | {100*row['liu_false_assurance_rate']:.2f}% | "
            f"{'yes' if row['liu_cor1_feasible'] else 'no'} |"
        )
    lines += [
        "",
        "Liu Algorithm 1 has no NO-GO state. Infeasible or high-FPR cases are recorded as ABSTAIN.",
        "Its theorem supplies a one-sided recall guarantee rather than a two-sided recall interval.",
        "The raw CSV therefore reports Liu's recall correction and FPR radius beside our interval width.",
        "",
        "## Sample-budget comparison",
        "",
        "| pi | ours empirical k* | Liu Cor. 1 minimum balanced n | Liu total sample count |",
        "|---:|---:|---:|---:|",
    ]
    for row in budgets:
        lines.append(
            f"| {row['pi']:.2f} | {row['ours_empirical_kstar_80pct_power_margin0.05']:,.0f} | "
            f"{row['liu_cor1_min_balanced_n_for_positive_q']:,} | {row['liu_total_clean_plus_mixture']:,} |"
        )
    lines += [
        "",
        "The first budget is the existing S0 empirical 80%-GO point for a 0.05 recall margin and a very large candidate pool.",
        "The Liu value is the smallest balanced n for which Corollary 1 permits q > 0 at target recall 0.8.",
        "These budgets answer related but non-identical questions; both show the same inverse-square rare-poison burden.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results_liu_matched_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    start = time.time()
    cells = []
    grid = [(f, p, m, n) for f in FAMILIES for p in PIS for m in MUS for n in SIZES]
    for index, (family, prevalence, mu, n) in enumerate(grid):
        seed = int(np.random.SeedSequence(20261010, spawn_key=(index,)).generate_state(1)[0])
        row = run_cell(family, prevalence, mu, n, args.replicates, seed)
        cells.append(row)
        print(f"[{index+1:02d}/{len(grid)}] {family} pi={prevalence} mu={mu} n={n}", flush=True)

    summary = summarize(cells)
    budgets = budget_rows()
    cells_path = args.output / "matched_cells.csv"
    summary_path = args.output / "matched_summary.csv"
    budget_path = args.output / "budget_comparison.csv"
    report_path = args.output / "REPORT.md"
    write_csv(cells_path, cells)
    write_csv(summary_path, summary)
    write_csv(budget_path, budgets)
    render_report(report_path, summary, budgets, args.replicates)
    manifest = {
        "paper": "Liu et al., JMLR 23(44), 2022",
        "algorithm": "Basic-CDF Algorithm 1 + Corollary 1",
        "seed_root": 20261010,
        "replicates_per_cell": args.replicates,
        "grid_cells": len(grid),
        "elapsed_seconds": time.time() - start,
        "files": {p.name: sha256(p) for p in (cells_path, summary_path, budget_path, report_path)},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
