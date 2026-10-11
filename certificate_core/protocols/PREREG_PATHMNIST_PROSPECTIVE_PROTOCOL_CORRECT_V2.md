# Prospective protocol-correct PathMNIST certificate v2

Freeze date: 2026-10-09 (Asia/Shanghai), before victim seed 8301 is opened.

## Purpose

This block closes the remaining provenance gap in the certificate manuscript.
Victim training, score construction, the trusted-member sampling key, registrar
logic, the six-field issuer interface, certificate decision rules, and independent
verification are all frozen before any outcome for seeds 8301--8305 is opened.

## Victims and attack

- Official PathMNIST 28x28 data with SHA-256
  `81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72`.
- Official-weight ResNet-18; all layers fine-tuned for five fixed epochs on Apple
  MPS with deterministic algorithms.
- Seeds 8301--8305, in order, without replacement, deletion, optional stopping,
  or seed-dependent tuning.
- Fixed initial weights alternate `_1`, `_2`, `_1`, `_2`, `_1`.
- BadNets all-to-one attack: target class 8, exactly 4,500 non-target members,
  solid white 4x4 bottom-right patch.
- Eligibility is balanced clean accuracy at least 0.75 and ASR at least 0.80.
  Every victim is reported even if ineligible.

## Frozen score experiment

The five continuous score families, directions, two-fold cross-fitting, reference
split, target population, and conventional descriptive endpoints are byte-for-byte
the same implementation used by the earlier PathMNIST score block: CF-CMD-U,
target-centroid, cross-fit spectral, activation clustering, and QUE score. This
block changes the formal seeds and certificate data flow, not the detector tuning.

Training/scoring seals `blind_scores.csv` separately from `private_truth.csv`.
The score file contains no poison truth. The truth file is private to the registrar
and post-seal verifier.

## Protocol-correct issuance

Before training, a fresh 32-byte key is generated and only its SHA-256 commitment
enters the freeze manifest. After score hand-off:

1. A truth-aware registrar authenticates a strict non-exhaustive pool of 3,000
   clean training members per victim-fold and uses the committed key for all draws.
2. The registrar exports exactly six public fields: `protocol`, opaque
   `trial_token`, `k`, `conformal_rank`, `candidate_count`, and `flagged_count`.
3. A separate issuer process receives only that public table and seals all
   GO/NO-GO/ABSTAIN decisions. It receives no scores, identifiers, detector/fold
   labels, conditions, roles, truth, truth hashes, pool membership, or key.
4. Only after the issuer manifest and decision hashes are sealed may the verifier
   open the key and private token map, independently replay every draw, and join
   truth for evaluation.

This is an application/process/schema separation, not an OS account, container,
or hardware-enclave claim.

## Certificate and endpoints

- Target recall 0.80; clean flag-rate cap 0.10; conformal alpha 0.05.
- Five detectors x two folds; Bonferroni delta 0.005 per detector-fold cell.
- Trusted budgets k = 120, 500, 2000; 200 draws per cell.
- Primary per-fold poison-count interval [1800,2700]; stress [1125,4500].
- Primary safety endpoint: valid primary false-GO count and rate, where false GO
  means GO while candidate recall is below 0.80 or clean flag rate exceeds 0.10.
- Failure recognition: among failed valid trials, the fraction receiving NO-GO or
  ABSTAIN; target at least 0.90.
- Utility: GO, NO-GO, and ABSTAIN rates at each k. Zero GO is safe but not useful.

## Falsification and reporting

Any valid detector-fold-budget cell whose one-sided 95% false-GO upper bound exceeds
its declared error budget is reported as unsupported/inconclusive. An invalid count
interval cannot support a safety claim. Zero observed events is never described as
zero residual risk. No detector, seed, budget, interval, endpoint, or eligibility
result may be removed after seed 8301 begins.
