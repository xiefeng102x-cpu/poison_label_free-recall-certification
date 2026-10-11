# Blind strong-architecture preregistration: PathMNIST / ResNet-50

Freeze date: 2026-10-07 (Asia/Shanghai), before victim, detector, certificate,
or test outcomes for seeds 2201--2205 were opened.

## Purpose and fixed comparison

This is a prospective architecture-transfer block for the detector-agnostic
certificate.  It changes the official victim from the previously evaluated
28x28 ResNet-18 to the official 28x28 ResNet-50 while holding the dataset,
preprocessing, BadNets attack, target, poison count, optimizer, epoch budget,
score families, count intervals, and certificate operating target fixed.

- Dataset: official PathMNIST, natural-frequency training split.
- Architecture: official MedMNIST small-image ResNet-50, 2048-dimensional
  penultimate representation, all parameters fine-tuned.
- Initializers: official `resnet50_28_1.pth` and `_3.pth`, alternated across
  formal seeds 2201--2205 as fixed in the JSON protocol.  This two-checkpoint
  design was fixed before any formal outcome was opened.
- Training: five epochs, Adam at 3e-4, batch size 128, fixed final checkpoint,
  deterministic PyTorch 2.7.1 on Apple MPS.
- Attack: exactly 4,500 uniformly selected non-target rows, class 8 target,
  solid white 4x4 bottom-right patch.
- Eligibility: balanced clean test accuracy >=0.75 and patched non-target ASR
  >=0.80.  Every formal seed is reported; there is no optional stopping.

The five score channels and their two-fold cross-fitting are exactly the same
as the frozen ResNet-18 block: CF-CMD-U, target-centroid distance,
cross-fitted spectral score, activation clustering, and QUE score.  The score
program stops at a truth-separated handoff.  Secret uniformly sampled verified
clean training members are drawn only after scores are fixed.

## Certificate and endpoints

Per detector-fold, alpha=0.05, target recall tau=0.80, clean flag-rate cap=0.10,
trusted-member budgets k in {120,500,2000}, and 200 draws are fixed.  The
primary per-fold poison interval is [1800,2700] and the stress interval is
[1125,4500].  Both the original per-detector allocation delta=0.025 and the
prespecified familywise allocation delta=0.05/(5 detectors x 2 folds)=0.005
will be reported.

Primary safety outcome: valid false-GO count/rate under the primary interval.
The familywise result is the paper-facing safety endpoint; the per-detector
allocation is retained for direct comparability and usability analysis.
Secondary outcomes are eligibility, detector recall/FPR, failure-recognition
rate, GO/ABSTAIN rates, and certified caught-poison lower bounds.

No seed, initializer, detector, score orientation, budget, count interval, or
outcome definition may be removed after outcomes are opened.
