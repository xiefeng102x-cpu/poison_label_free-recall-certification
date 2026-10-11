# Liu Algorithm 1 matched baseline

This run implements the Basic-CDF Algorithm 1 and Corollary 1 from Liu et al. (JMLR 2022).
Both procedures receive the same clean and mixture score tables. Exact prevalence is supplied to both.
Each cell uses 1000 replications. The common operating target is recall >= 0.80 and FPR <= 0.10.
Only cells whose nominal 5% FPR threshold has recall >= 0.85 enter the power summary.

## Matched joint decision

| pi | n clean = n mixture | strong cells | ours GO | Liu GO | ours false assurance | Liu false assurance | Liu Cor. 1 feasible |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.05 | 2,000 | 3 | 2.57% | 0.00% | 0.00% | 0.00% | no |
| 0.10 | 2,000 | 3 | 12.47% | 0.00% | 0.00% | 0.00% | no |
| 0.20 | 2,000 | 3 | 44.63% | 0.00% | 0.00% | 0.00% | no |
| 0.05 | 10,000 | 3 | 21.43% | 0.00% | 0.00% | 0.00% | no |
| 0.10 | 10,000 | 3 | 56.00% | 0.00% | 0.00% | 0.00% | no |
| 0.20 | 10,000 | 3 | 93.23% | 53.30% | 0.00% | 0.00% | yes |

Liu Algorithm 1 has no NO-GO state. Infeasible or high-FPR cases are recorded as ABSTAIN.
Its theorem supplies a one-sided recall guarantee rather than a two-sided recall interval.
The raw CSV therefore reports Liu's recall correction and FPR radius beside our interval width.

## Sample-budget comparison

| pi | ours empirical k* | Liu Cor. 1 minimum balanced n | Liu total sample count |
|---:|---:|---:|---:|
| 0.01 | 1,840,626 | 2,509,152 | 5,018,304 |
| 0.02 | 410,046 | 621,000 | 1,242,000 |
| 0.05 | 62,252 | 96,372 | 192,744 |
| 0.10 | 14,490 | 22,874 | 45,748 |
| 0.20 | 3,570 | 5,133 | 10,266 |

The first budget is the existing S0 empirical 80%-GO point for a 0.05 recall margin and a very large candidate pool.
The Liu value is the smallest balanced n for which Corollary 1 permits q > 0 at target recall 0.8.
These budgets answer related but non-identical questions; both show the same inverse-square rare-poison burden.
