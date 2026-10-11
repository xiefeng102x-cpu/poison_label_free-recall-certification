# Preregistration: certificate-aware red-team and multi-family / multi-dataset breadth

Freeze date: 2026-10-07 (Asia/Shanghai). Frozen by recording its SHA-256 before
the first victim in either part is trained. No change may depend on results;
deviations go to the deviation log. Victim training, detector runs, and scoring
are executed by the user's team. This file specifies the threat models,
certificate parameters, experiment matrices, falsifiable predictions, and
stopping rules only.

This protocol targets the two empirical gaps left open by the manuscript:
(Part A) the certificate-aware adversary of Corollary "Robustness to a
certificate-aware adversary", whose two avenues attack the mechanism rather
than the scores; and (Part B) attack-family and dataset breadth, so the
safety claim is not tied to patch-style BadNets on a single modality.

---

## Part A. Certificate-aware red-team

The prior adaptive block (PREREG_BLIND_ADAPTIVE_MULTISCORE_V1) attacks the
*scores* by representation alignment. Part A instead attacks the *certificate
mechanism itself*, along the exactly two avenues the Corollary identifies as
lying outside the protected randomness. The point is to show empirically that
(a) when the secret draw is protected the guarantee holds, and (b) the secret
draw is genuinely load-bearing, so the assumption is not vacuous.

### A1. Member-selection avenue (observing or biasing the secret draw)

Threat model, abstract level only. The attacker knows the detector, the score
map, the certificate rule and its parameters, the declared count range, and the
public calibration *procedure*. The attacker may attempt, by signals available
at training time, to anticipate or steer which clean samples will later serve as
trusted members. The attacker never sees a realized formal draw. No attack
construction code is specified here; the red-team is defined by what the
certificate must survive, not by a recipe.

Two pre-registered conditions, run as a matched pair:
- **Protected (primary).** Trusted members are drawn by the defender uniformly
  at random from a pool the attacker cannot enumerate before training, with the
  score map frozen before the draw. This instantiates the Corollary's
  hypothesis.
- **Leaked (ablation, safety-critical control).** The protocol deliberately
  reveals the realized member identities to the attacker's selection process
  before the draw is used, violating Assumption "Secret exchangeable
  calibration". This condition is expected to break the guarantee and exists to
  demonstrate that the secret draw is the load-bearing element, not to claim a
  defended setting.

Falsifiable predictions:
1. In the Protected condition the valid false-GO rate is <= delta (Corollary).
   Any valid familywise false GO here falsifies the robustness claim.
2. In the Leaked condition the false-GO rate exceeds delta. If it does *not*
   rise, the paper must weaken its claim that secrecy of the draw is necessary,
   because the assumption would then appear non-load-bearing.

### A2. Pool-contamination avenue (Proposition "Contaminated calibration")

The attacker injects a fraction epsilon of poison into the trusted pool.
Pre-registered grid epsilon in {0, 0.01, 0.02, 0.05, 0.10}, five seeds each.

Falsifiable predictions, directly from Prop (contamination is one-sided safe):
1. **GO stays safe.** The valid false-GO rate remains <= delta for every
   epsilon. A GO manufactured by contamination falsifies the one-sided-safety
   claim.
2. **NO-GO power degrades monotonically.** Failure-recognition (NO-GO) rate on
   truly failing detectors decreases as epsilon grows. A non-monotone or flat
   curve would contradict the stochastic-dominance argument and must be
   reported as such.

---

## Part B. Multi-family, multi-dataset breadth

All sub-blocks reuse the frozen five-score, two-fold interface, the certificate
operating point, and the truth-separated handoff. Only the dataset and the
attack family vary. Each (dataset x family) is an independent blind sub-block
with its own five formal seeds, its own eligibility gate (balanced clean
accuracy >= 0.75 and ASR >= 0.80 on that family's trigger), and no optional
stopping.

- Datasets: CIFAR-10, CIFAR-100, GTSRB, and BloodMNIST (medical modality held
  in for cross-modality coverage). Architecture fixed to the frozen ResNet-18
  so family effects are not confounded with architecture.
- Attack families (named published constructions; the user's team implements
  them from their original papers, not from this file):
  - BadNets patch (reference/anchor family, already evaluated).
  - Blended-injection (Chen et al., 2017).
  - WaNet warping-based trigger (Nguyen and Tran, 2021).
  - Label-consistent / clean-label (Turner et al., 2019).
  - A feature-space / optimized-trigger family (e.g. Narcissus-style clean-label
    optimization), as the hardest case for representation-based scores.

### Certificate and endpoints (both parts)

Per detector-fold: alpha = 0.05, tau = 0.80, clean flag-rate cap = 0.10,
k in {120, 500, 2000}, 200 draws, fpr_mode = "tight". Primary and stress poison
intervals as in the primary blind block, scaled to each dataset's poison count
and recorded per sub-block in the JSON protocol before outcomes are opened.
Familywise allocation: delta = 0.05 divided by the full count of
(condition x dataset x family x detector x fold) cells actually run; this
familywise number is the paper-facing safety endpoint, and the per-detector
delta = 0.025 is retained for comparability.

Primary safety outcome across every cell: valid false-GO rate, where a false GO
is any GO for a valid interval while realized recall < 0.80 or realized clean
flag rate > 0.10. Secondary outcomes: eligibility, detector recall/FPR,
failure-recognition rate, GO rate, certified caught-poison lower bound, and,
for Part A2, the NO-GO-vs-epsilon curve.

### Global falsification and reporting discipline

The breadth claim is that the certificate's false-GO guarantee holds uniformly
across every attack family, dataset, and red-team condition at the familywise
budget, while utility (GO power) is permitted to vary and even to collapse to
universal abstention on the hardest families. A zero-false-GO, high-abstention
outcome supports safety while showing a utility limit; both must be reported.
An attack that evades a detector is never relabeled inconclusive merely because
its ASR is below a preferred value: viability is judged only by the frozen
eligibility gate, and raw accuracy/ASR are always shown. No seed, condition,
epsilon value, dataset, family, detector, budget, interval, or endpoint may be
removed or redefined after formal outcomes are opened.
