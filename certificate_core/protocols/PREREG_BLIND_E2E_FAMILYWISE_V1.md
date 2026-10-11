# Pre-outcome familywise analysis for the blind five-detector block

Declared on 2026-10-07 after the formal training command started but before any
victim, detector, or certificate outcome was inspected.  This is a separate
pre-outcome sensitivity analysis; it does not modify or replace the hash-frozen
per-detector primary analysis in `BLIND_E2E_FREEZE_MANIFEST_V1.json`.

The score maps, folds, trusted-member draws, thresholds, candidate tail counts,
budgets, poison-count intervals, and 200 draws are identical to the primary blind
block.  Only the error allocation changes:

- detector count D = 5;
- fold count F = 2;
- family error budget = 0.05;
- each detector-fold certificate receives delta = 0.05/(D F) = 0.005.

This makes the probability that any GO selected from the ten prespecified
detector-fold certificates is false at most 0.05 by the union bound.  Results must
report the additional abstention cost.  The analysis reads the already sealed
truth-free sufficient statistics and recomputes decisions; it may not change score
orientation, trusted draws, or count intervals.
