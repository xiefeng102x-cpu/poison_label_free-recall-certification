# Preregistration: CIFAR-10 / VGG16-BN fully blind certificate block

Status before formal seeds: **frozen before execution of seeds 4201--4205**.

## Purpose

This is the one planned orthogonal full-blind block. It changes both dataset
(PathMNIST to native CIFAR-10) and architecture family (ResNet to VGG16-BN)
while retaining a standard all-to-one BadNets trigger. It tests transport of the
auditability certificate; it is not a claim of universal backdoor detection.

The certificate-aware adaptive PathMNIST gate completed before this protocol
was opened for formal execution. Its preregistered familywise result contained
zero valid false-GO decisions, so the predeclared stop condition did not fire.

## Development-to-freeze rule

Seed 4200 is nonformal and may expose only clean BCA and ASR. Detector scores,
poison-screening TPR/FPR, certificate decisions, and poison-truth joins are
forbidden during development. The training endpoint is the earliest one among
epochs 10, 15, 20, 25, and 30 with BCA at least 0.80 and ASR at least 0.90.
Epoch 10 was the first viability endpoint and produced BCA 0.8064 and ASR
0.9550. Before any formal seed, detector score, or certificate decision was
opened, the victim-fidelity requirement was strengthened to BCA at least 0.90
because a barely 80% CIFAR victim would not constitute the intended strong-model
test. Epoch 25 was the first prelisted checkpoint meeting the stronger target
(BCA 0.9015, ASR 0.9723). The formal endpoint is therefore 25 epochs with the
original 30-epoch cosine horizon. The dated preformal amendment records this
change; it will not be revisited using formal outcomes.

## Frozen experimental unit

- Formal seeds: 4201--4205, with no replacement.
- Data: the hash-checked official CIFAR-10 Python release.
- Split: fixed stratified 45,000-row training and 5,000-row validation split;
  the official 10,000-row test split is untouched by fitting.
- Victim: fresh CIFAR-native VGG16-BN, 14,724,042 trainable parameters and a
  512-dimensional penultimate representation, trained for 25 epochs.
- Attack: 2,250 uniformly sampled non-target training rows receive a solid
  white 4x4 bottom-right patch and label 8, exactly 5% of training rows.
- Eligibility: BCA at least 0.75 and ASR at least 0.90. All five seeds are
  reported even if a victim is ineligible.
- Score families: CF-CMD-U, target-centroid distance, cross-fit spectral,
  activation clustering, and QUE score. No detector is selected after seeing
  outcomes.

## Three-stage blind handoff

1. The victim runner writes detector scores, a restricted preverified clean
   member pool, and a separately hashed private truth file. It does not issue a
   certificate conclusion.
2. The certificate runner reads only scores and the trusted-member pool. It
   seals GO/NO-GO/ABSTAIN decisions without opening the private truth file.
3. The independent verifier checks all hashes, replays every calibration draw,
   and only then joins poison truth to calculate false assurance.

The score runner, certificate runner, verifier, model definition, score-family
implementation, certificate library, configuration, and this preregistration
must all match the freeze manifest before seed 4201 starts.

## Certificate and endpoints

The target operating point is recall at least 0.80 and target-clean flag rate
at most 0.10. The detector threshold uses upper-tail conformal rank at nominal
alpha 0.05. Trusted-member budgets are 120, 500, and 2,000, with 200 fixed draws
per cell. The primary poison-count interval per fold is [900, 1350]; [560, 1374]
is a reported lower-prevalence stress interval. The latter upper endpoint is
bounded by the smallest candidate set remaining at the largest trusted budget.

The primary error allocation is familywise from the outset: delta_family=0.05
is divided across five detectors and two cross-fit folds, giving
delta=0.005 per detector-fold. A detector-level deployment decision is GO only
when both folds are GO, NO-GO only when both folds are NO-GO, and ABSTAIN
otherwise. The primary endpoint is the count and rate of valid false-GO
deployment decisions under the primary interval. GO, NO-GO, ABSTAIN,
failure-recognition, conventional detector performance, BCA, and ASR are
secondary and will all be reported.

## Stopping and interpretation

- Abort before outcomes on any dataset, source, configuration, or hash mismatch.
- Do not replace seeds, tune scores, change intervals, or change budgets after
  opening any formal score or decision.
- An ineligible victim weakens the victim block and is reported; it is not
  silently removed or replaced.
- A false-GO is a certificate safety failure. High abstention is a utility cost,
  not evidence of safety failure and not a reason for post hoc retuning.
- This block can support transport to one new dataset/architecture pair only.
  It cannot establish universality over datasets, attacks, or architectures.
