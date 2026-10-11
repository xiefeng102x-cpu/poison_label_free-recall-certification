# Blind end-to-end preregistration: PathMNIST / ResNet-18 / multi-detector certificate v1

Freeze date: 2026-10-07 (Asia/Shanghai), before any victim, detector, or certificate
output for seeds 1201--1205 was opened.

This is the first paper14 block in which the victim seeds, detector score maps,
trusted-member draw, poison-count ranges, and GO/NO-GO/ABSTAIN analysis are frozen
before outcomes are available.  Earlier PathMNIST and GTSRB blocks remain valuable
prospective victim experiments, but their certificate handoff was post-endpoint.

## Fixed victim experiment

- Dataset: official PathMNIST (`pathmnist.npz`), natural-frequency training split.
- Architecture: official-weight 28x28 ResNet-18, all parameters fine-tuned.
- Formal seeds: 1201, 1202, 1203, 1204, 1205; ordered, no substitutions and no
  optional stopping.
- Initial weights alternate `resnet18_28_1.pth`, `resnet18_28_2.pth`, beginning
  with `_1` at seed 1201.
- Training: five epochs, Adam, learning rate 3e-4, batch size 128, no scheduler or
  augmentation, fixed final checkpoint.  Only the suspect victim is trained; a
  paired clean model is not needed by the certificate and is outside this block.
- Attack: BadNets all-to-one, target class 8, exactly 4,500 uniformly selected
  non-target training rows, solid white 4x4 bottom-right patch.
- Victim eligibility: balanced clean test accuracy >= 0.75 and patched non-target
  ASR >= 0.80.  Every seed is reported even when ineligible.
- Device: Apple MPS, PyTorch 2.7.1, deterministic algorithms.  CPU substitution
  after opening a formal seed is forbidden.

## Frozen score interface

The suspect victim emits 512-D penultimate features.  Candidate rows are all
training members assigned to target class 8 after poisoning.  Two folds are fixed by
`numpy.default_rng(20301000 + victim_seed)`.  Each score map is fitted on the
opposite fold, then applied to the scoring fold and untouched target-class test
rows.  Higher values always mean more suspicious.

Five detector families are fixed:

1. `cf_cmd_u`: raw-space opposite-half mean direction relative to 512 trusted
   target-class validation reference rows.
2. `target_centroid`: Euclidean distance from that reference centroid.
3. `crossfit_spectral`: squared projection onto the leading centered singular
   direction of the opposite candidate half.
4. `activation_clustering`: reference-whitened PCA(16) plus two-means fitted on
   the opposite half; the component farther from the reference mean is suspicious.
5. `que_score`: reference-whitened quantum-entropy weighted covariance score,
   fitted on the opposite half with temperature 4.

These are continuous score channels for certificate testing.  The names do not
claim byte-for-byte equivalence to a full detector's model-training pipeline.
SCAn's class-level statistic and SPECTRE's binary retained mask are not forced into
an artificial continuous ranking and are therefore outside this score-certificate
block.

The fixed validation calibration split supplies only descriptive conventional
threshold endpoints.  Certificate calibration instead uses secret uniformly random
verified-clean *training members* selected after every score has been fixed.

## Information firewall and frozen certificate

The training/scoring program writes:

- `blind_scores.csv`: scores and roles, without poison truth;
- `private_truth.csv`: the evaluation truth in a separate file;
- content-hashed checkpoints, feature archives, split hashes, and receipts.

The certificate stage may use `private_truth.csv` only (i) to simulate the
defender's independent verification that selected trusted members are clean and
(ii) after decisions are fixed, to evaluate realized recall, clean flag rate and
false GO.  Detector fitting, score orientation, thresholds, and declared count
ranges may not use truth.

Fixed certificate settings per victim, detector and fold:

- alpha = 0.05, target recall tau = 0.80, clean flag-rate cap = 0.10;
- familywise fold allocation delta = 0.05 / 2;
- trusted-member budgets k in {120, 500, 2000};
- 200 secret draws per cell;
- exact randomized tie breaking with seed fixed by the cell and draw;
- tight realized clean-rate bound;
- primary declared per-fold poison-count interval [1800, 2700];
- stress interval [1125, 4500].

The intervals are defense-side declarations fixed from the attack design before
the realized fold counts are opened.  If a realized fold count lies outside an
interval, that interval is marked assumption-invalid and cannot support a safety
claim.

## Endpoints and falsification rules

Primary safety endpoint: false-GO rate under the primary interval, where false GO
means GO while realized recall is below 0.80 or realized clean flag rate exceeds
0.10.  A detector/fold/budget cell falsifies the claimed 5% safety tolerance if its
one-sided 95% binomial upper confidence bound is above 0.05 and the excess is not
explained by an out-of-range declared interval.

Failure recognition: for cells whose realized recall is below 0.80, report the
fraction of decisions that are NO-GO or ABSTAIN; the preregistered target is >=90%.
Usability is secondary: GO rate, ABSTAIN rate, certified caught-poison lower bound,
and human verification budget.  Zero GO is safe but may be useless and must never be
presented as detector success.

No detector, seed, score orientation, budget, interval, or outcome definition may
be removed after results are opened.  Any amendment must be timestamped, hashed,
and reported as post-unblinding.
