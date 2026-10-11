# Reproduction commands

Run from the repository root:

```bash
cd <USER_HOME>/Desktop/study/tex/Blackdoor
```

The recorded environment used:

```text
Python: <USER_HOME>/miniforge3/envs/quantum_clean/bin/python
BloodMNIST: <TMP>/blood-ca-data/bloodmnist.npz
PathMNIST: <TMP>/paper14-data/pathmnist.npz
```

The machine-readable evidence-host snapshot is
`results/top_journal_program_v1/reproducibility_environment_2026-10-02.json`.
`requirements-replay.txt` pins the top-level Python interfaces used by the
archived replays and tests.  A clean reconstruction can begin with:

```bash
python3.9 -m venv .venv-paper14
.venv-paper14/bin/python -m pip install -r Blackdoor_1/paper14/requirements-replay.txt
```

The requirements file is deliberately labelled a top-level replay lock, not a
fully resolved cross-platform transitive lock. Exact bit identity is claimed
only where an artifact receipt demonstrates it on the recorded macOS arm64/MPS
host. Fresh MedMNIST acquisition and selected historical BloodMNIST runners
also require MedMNIST 3.0.2; archived NPZ verification does not import it.

Third-party terms and the nonfinal bilingual submission declarations are
separated from scientific claims in `THIRD_PARTY_NOTICES.md` and
`SUBMISSION_STATEMENTS_DRAFT.md`. The historical MNIST raw-resource and direct
Phase-A input chain is recorded in
`mnist_source_provenance_receipt_2026-10-02.json`. Its four gzip files match
torchvision's official MD5 constants and their SHA-256 values are recorded;
the original generation-time torchvision version is not recoverable from the
retained evidence. `MNIST_HISTORICAL_ENVIRONMENT_RECOVERY_AUDIT_2026-10-03.md`
prevents back-projection of the current replay pin and closes the release
action by excluding raw MNIST files while retaining the hash-bound direct input
chain.

The PathMNIST benchmark-weight release decision is recorded separately in
`PATHMNIST_WEIGHT_REDISTRIBUTION_AUDIT_2026-10-02.md` and its machine receipt.
The Zenodo record is the acquisition source, but its displayed license value is
blank; reproducibility packages therefore publish the DOI, archive/member
hashes, selection map, runners and regeneration instructions rather than
redistributing official or official-initialized derived weight binaries unless
explicit authorization is later archived.

The checked-in preterminal release/public-profile manifests describe the last
complete staged snapshot.  They are intentionally stale after P6-F and the
TIFS main/supplement were added.  Do not present them as current and do not
rebuild them merely to obtain a temporary green check while PathMNIST is
sealed.  After terminal Path integration, rebuild the release payload with:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/build_release_archive_manifest.py
```

The generated manifest is an inventory and risk audit, not a release archive.
It includes the complete non-`.git` BackdoorBench snapshot in its checksum
closure, while raw benchmark archives remain excluded.

Then rebuild the separate complete public-code profile inventory:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/build_public_code_profile.py
```

The counts below describe the historical preterminal snapshot and are retained
for provenance: 611 files, including 153 project-owned source, test,
configuration and theory files, 35 frozen protocol/contract files and 411
BackdoorBench files. The current public release is larger and uses the
repository-level MIT License for project-owned code. BackdoorBench retains its
separate terms.

Audit the deterministic blind-review path substitutions without writing a
package:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/build_blind_review_package.py \
  --audit-receipt Blackdoor_1/paper14/results/top_journal_program_v1/blind_review_sanitization_audit_preterminal_2026-10-02.json
```

After rebuilding the manifest, audit it before staging. A preterminal
end-to-end packaging rehearsal must then be created outside the source tree and
verified independently:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/build_blind_review_package.py \
  --stage --destination <STAGING_DIR> \
  --ack BUILD_PATH_SANITIZED_BLIND_REVIEW_COPY
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_blind_review_package.py <STAGING_DIR>
```

The independent verifier rejects missing, extra, tampered or wrongly excluded
files, recomputes every staged hash and scans readable text again for residual
paths. The preterminal rehearsal is a packaging QA check; the submission copy
must still be rebuilt and replayed after the terminal PathMNIST route is
integrated. The current rehearsal contains 839 files (837 payload plus two
control files), excludes four license-ambiguous weight binaries, and runs the
171-test reviewer profile as 166 passes plus five explicit policy skips. The audit
receipt is control-plane metadata outside the release payload, avoiding a
manifest/receipt hash cycle. The guarded `--stage` mode is intentionally
isolated from the evidence workspace and emits a transformation ledger in the
user-selected staging directory.

## Lightweight final replay

This does not retrain victims. It independently reconstructs confusion matrices and pooled summaries from saved rows, verifies protocol hashes and earlier receipts, and checks the final artifact bundle.

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_top_journal_final_bundle.py
```

Expected output: `status: PASS` and a refreshed `results/top_journal_program_v1/final_bundle_receipt.json`.

Pinned BackdoorBench addendum replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_official_backdoorbench_confirmation.py
```

Expected output: `status: PASS` and a refreshed `official_backdoorbench_confirmation_receipt.json`.

Checkpoint-auditability theory probe replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_checkpoint_auditability_theory_probe.py
```

Expected output: `status: PASS`, 36 replayed view configurations and 18 paired comparisons.

Realized auditability-geometry diagnostic replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_auditability_geometry_diagnostic.py
```

Expected output: `status: PASS`, 36 replayed Phase-A configurations, 30
reconstructed BloodMNIST victims, 48 population summaries and five paired
contrasts.  This command performs feature reconstruction and statistical
replay only; it does not train victims or read PathMNIST artifacts.

Candidate-independent single-sample reference audit replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_single_sample_reference_audit.py
```

Expected output: `status: PASS`, 30 reconstructed BloodMNIST victims, 25,800
replayed query scores/decisions, six population summaries and the frozen
`NO_RELIABLE_SINGLE_SAMPLE_CHANNEL_IN_THIS_ARCHIVE` decision.  This is a
no-training retrospective replay and does not read PathMNIST artifacts.

Bilingual manuscript claim/evidence consistency audit:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_manuscript_claim_consistency.py
```

Expected output: `PASS`, seven recomputed central claim families, 74 required
bilingual claim/configuration tokens (including 26 frozen-configuration
tokens), seven claim-matrix decisions and eight forbidden positive forms.  The
receipt explicitly records `pathmnist_partial_artifacts_read: false`.

Preterminal TIFS main/supplement build and verification:

```bash
cd Blackdoor_1/paper14/manuscript_tifs_en
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output main_tifs.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output supplement_tifs.tex
cd ../../..
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_tifs_preterminal_manuscript.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/pytest -q \
  Blackdoor_1/paper14/tests/test_tifs_preterminal_manuscript.py
```

Expected verifier status: `PASS_PRETERMINAL`, 9 main-paper pages, 7
supplement pages, 22/22 main citations, 7/7 supplement citations, eight formal
proposition headings, nine frozen configuration protocols bound through 35
ledger assertions, zero overfull boxes and a matching hash-bound visual-QA
receipt.  This command never reads sealed PathMNIST performance endpoints and never promotes
the paper to submission-ready.  Any source/PDF change invalidates the visual
receipt and requires rendering and inspecting every page again.

Eight-class BackdoorBench development/confirmation replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_multiclass_backdoorbench_program.py
```

Expected output: `status: PASS`, 191,344 development sample rows and 478,360 confirmation sample rows.

AGPD feature-adapter development/confirmation replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_bloodmnist_multiclass_agpd_adapter.py
```

Expected output: `PASS`, 23,918 development flags and 59,795 confirmation flags, with pooled confirmation TPR/FPR `0.053333/0.002659`.

Fresh-query calibration development/confirmation replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_bloodmnist_fresh_query_calibration.py
```

Expected output: `PASS`, 11,984 development and 29,960 confirmation query rows. Confirmation member/fresh FPR is `0.040687/0.031532` for suspect hidden64 and `0.043090/0.054354` for paired drift.

CF-CMD-U trusted-reference/calibration budget replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_reference_calibration_budget_sensitivity.py
```

Expected output: `PASS`, 15 budget cells, 150 seed-grid metrics and 99,000
sample decisions.  The verifier independently reconstructs trusted splits,
cross-fit directions, scores, rank p-values, flags, pooled summaries and
10,000-draw seed-block intervals from archived feature caches, and exactly
recovers the original 120/80 CF-CMD-U endpoint.  This is a no-training
retrospective sensitivity replay and does not read PathMNIST artifacts.

Official PathMNIST acquisition replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_acquisition.py
```

Expected output: `status: PASS`, official MD5
`a8b06965200029087d5bd730944a56c1`, six verified arrays, and all official
split/class counts. The dataset itself remains outside the repository at the
path recorded above.

Official PathMNIST ResNet-18 checkpoint provenance and clean sanity replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_official_weights.py
```

Expected output: `status: PASS`, two strict-loaded 11,173,449-parameter
checkpoints, matching archived CRC32/MD5/SHA-256 values, and clean official-split
metrics.  This command verifies provenance and architecture compatibility only;
it is not attack or detector evidence.

PathMNIST lightweight external-domain replays:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_external_validation.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_external_validation_v2.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_warmstart_pilot.py
```

Each verifier is expected to return `status: PASS` and replay 324,000
method-level decisions. Across V1, V2 and the warm-start pilot this is 972,000
decisions. PASS means that the archived evidence is internally reproducible;
it does not mean that the external-domain confirmation gate passed.

## Standard PathMNIST victim and detector routes

The official-weight five-epoch victim route completed seeds 93/94 and verifies
independently; it is victim-feasibility evidence only. The 100-epoch
from-scratch route remains prepared but unrun.

Full-data from-scratch official-style ResNet-18 (100 epochs, seeds 93 and 94):

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/run_pathmnist_resnet18_victim_pilot.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_resnet18_victim_pilot.py
```

Official-weight full-parameter warm start replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/run_pathmnist_official_weight_warmstart.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_official_weight_warmstart.py
```

The from-scratch dry run archived under
`results/pathmnist_resnet18_victim_pilot_dry_run_official_compat/` must verify as
`DRY_RUN_VALID_NONFORMAL`. Warm-start victim results do not replace detector
confirmation.

The real-input MPS smoke can be independently replayed on the Apple host:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_official_weight_detector_confirmation.py \
  --smoke-receipt Blackdoor_1/paper14/results/pathmnist_real_input_mps_smoke_v1/smoke_receipt.json \
  --dataset <TMP>/paper14-data/pathmnist.npz
```

Expected status is `PASS_NONFORMAL_REAL_INPUT_MPS`. The fixed 103--107 formal
block has produced all five clean/suspect model pairs, feature archives,
per-seed scores and terminal aggregate files. Do not launch, resume or replace
any formal seed. The first direct terminal replay exposed a path-resolution
incident: repository-root-relative feature paths were incorrectly prefixed by
the verifier's output directory even though every actual feature SHA-256
matched its receipt. The original verifier and all formal artifacts remain
unchanged. Replay through the amendment-bound, symlink-only staging view before
inspecting or reporting any detector endpoint:

```bash
cd Blackdoor_1/paper14
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  src/run_pathmnist_terminal_staged_replay.py
```

The first symlink-staged replay returned from the frozen verifier but the host
removed the temporary dataset before the wrapper could persist its separate
receipt.  That process therefore supplies no scientific verdict.  The frozen
V2 amendment binds the archive to the original acquisition receipt and formal
manifest before replay and removes only the post-run dataset reopen:

```bash
cd Blackdoor_1/paper14
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  src/run_pathmnist_terminal_staged_replay_v2.py
```

The only acceptable formal replay status is `PASS` with
`formal_evidence: true`, no failures, verified seeds exactly 103--107 and 25
verified method rows. The wrapper first verifies the hashes of the frozen
amendment, predecessor wrapper, unchanged verifier and its own source; it then
destroys its temporary view and writes
`results/top_journal_program_v1/pathmnist_terminal_staged_replay_v2_receipt_2026-10-04.json`.
It never edits or copies a replacement evidence set. A clean verifier exit
alone is insufficient because the nonformal dry-run status also exits
successfully. See
`PATHMNIST_TERMINAL_REPLAY_PATH_INCIDENT_2026-10-03.md` and
`PATHMNIST_TERMINAL_REPLAY_DATASET_LIFETIME_INCIDENT_2026-10-04.md` for the two
incidents and their exact no-scientific-change boundaries.

Only after the V2 receipt exists, select the frozen terminal route:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/select_pathmnist_terminal_route_v2.py
```

The selector validates the full V2 receipt and file hashes before parsing the
formal manifest.  Any control failure selects R0 without opening performance
fields; a complete PASS mechanically selects R1, R2 or R3 and writes a
hash-bound route receipt before any manuscript integration.

Preterminal final-bundle migration readiness can be checked without reading a
PathMNIST endpoint:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_final_bundle_v2_migration_preterminal.py
```

Expected status is `PASS_PRETERMINAL_MIGRATION`.  This preserves the old final
verifier and receipt as explicitly stale history, validates the V2 inventory
and current manuscript/TIFS control values, and remains `submission_ready:
false`.  It is not a substitute for the post-route full regression and V2
final-bundle verifier.

The post-selection execution surface can be checked separately without
opening the V2 replay receipt, route receipt, formal manifest or metrics:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_terminal_integration_execution_contract_v2.py
```

Expected status is `PASS_PRETERMINAL_EXECUTION`: all four frozen routes retain
their pooling permissions; 19 mutable targets, four protected historical
artifacts, two terminal inputs, six new receipts and ten ordered closure steps
are covered.  In particular, the historical V1 final-bundle verifier and
receipt must never be rewritten as current.  The receipt also verifies 19
preterminal mutable-surface hashes and four protected-history hashes; terminal
closure must change every former hash and preserve every latter hash.

The prepared V2 final-bundle verifier has a non-outcome self-check mode:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_top_journal_final_bundle_v2.py \
  --mode preterminal-self-check
```

Expected status is `PASS_PRETERMINAL_FINAL_VERIFIER_PREP`, with both
`pathmnist_terminal_receipts_opened` and
`pathmnist_performance_artifacts_parsed` false.  Do not run `--mode terminal`
as a polling command.  It is authorized only after the V2 replay receipt,
frozen route selection, all 19 surface updates and the five other terminal
receipts exist; terminal mode then writes the final output, never the V1
receipt.

After a valid route receipt exists, build the single route-authorized claim
capsule before editing any manuscript or evidence table:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/build_pathmnist_terminal_claim_capsule_v2.py
```

R1 capsules contain all five victim rows but no pooled detector summary. R2/R3
capsules contain the primary plus four frozen baselines, per-seed rows and
seed-block intervals, together with bounded English/Chinese result sentences
and the exact route marker. The capsule is invalid unless the replay and route
receipts pass the same control gate as the V2 final verifier.

Independently replay the capsule before using it for any manuscript edit:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_pathmnist_terminal_claim_capsule_v2.py
```

This verifier recomputes the five victim rows, every route-permitted detector
summary and interval, the point gate, source hashes and both bounded claim
sentences. Its PASS receipt is a separate terminal output; the capsule
generator cannot self-certify.

The post-route TIFS verifier can be prepared without terminal inputs:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_tifs_terminal_manuscript_v2.py \
  --mode preterminal-self-check
```

The preparation receipt must report 35 configuration tokens, nine frozen
protocol hashes and no terminal-receipt or performance access. After route
integration, recompilation and renewed visual inspection, run the same command
with `--mode terminal` to create `tifs_main_terminal_receipt_2026-10-04.json`.
The historical preterminal verifier and visual-QA receipt remain unchanged.

Current terminal PathMNIST/TIFS state can be replayed without rerunning victim
training:

```bash
cd Blackdoor_1/paper14
python src/verify_pathmnist_terminal_claim_capsule_v2.py
python src/verify_tifs_terminal_manuscript_v2.py --mode terminal
```

Expected statuses are `PASS` and `PASS_TERMINAL`. The selected route is
`R3_ELIGIBLE_PRIMARY_GATE_PASS`; all five detector-confirmation victims are
eligible and the primary endpoint is 99.79% TPR / 0.54% all-clean FPR. These
commands verify existing artifacts and receipts; they do not authorize a new
seed, threshold, victim retraining, or outcome-dependent rerun.

After all claim surfaces and source/PDF hashes are synchronized, terminal
package closure runs in this order:

```bash
python src/run_full_source_regression_terminal_v2.py --mode terminal
python src/build_terminal_release_profile_v2.py --mode prepare-manifests
python src/run_terminal_reviewer_package_v2.py --mode terminal \
  --destination /path/to/empty/reviewer-staging-directory
python src/build_terminal_release_profile_v2.py --mode finalize
python src/verify_top_journal_final_bundle_v2.py --mode terminal
```

The staging destination must be a non-existent path outside the source tree;
the builder creates it and rejects even a pre-created empty directory. Each
step consumes the preceding PASS receipt. The
V1/preterminal receipts are immutable history and must not be overwritten.

The preserved 2026-10-04 terminal chain is closed with `PASS_TERMINAL_SOURCE_REGRESSION`
(241/241), `PASS_TERMINAL_REVIEWER_PACKAGE` (236 pass and five frozen policy
skips), `PASS_TERMINAL_RELEASE_PROFILE`, and `PASS_TERMINAL_BUNDLE`. These
statuses establish mechanical reproducibility, not author licence, accession,
declaration, submission, or publication authorization.

After the P6-D v4/v5 integration, the current workspace separately passes
250/250 tests as recorded in `full_source_regression_current_2026-10-06.json`.
This does not mutate or silently extend the older hash-bound release inventory.
Regenerate the public/reviewer profiles and final-bundle receipt before packaging
the current manuscript snapshot.

Theory assumption/non-implication surface consistency:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_theory_claim_surface_consistency.py
```

Expected status is `PASS`: eight proposition headings on each formal surface,
52 theory-boundary anchors, five repaired hostile-audit interfaces, 12
positive-overclaim checks, and `pathmnist_artifacts_read: false`.

Threat-model and information-budget surface consistency:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_threat_model_surface_consistency.py
```

Expected status is `PASS`: five claim surfaces, 50 threat-model/boundary
anchors, five privilege-ledger rows, 25 positive-overclaim checks, and
`pathmnist_artifacts_read: false`.  The contract keeps attacker knowledge
separate from attacker control, preserves the trusted-root and optional clean-
checkpoint assumptions, and prevents a candidate-batch rule from being
silently rewritten as a context-free single-sample detector.

## P6-D incidents, v3/v4 replay, and v5 development stop

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6d_v1_formal_failure.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6d_v2_formal_failure.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/run_p6d_final_observable_adaptive_v3.py self-check
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6d_v3_confirmation_delivery.py \
  --output Blackdoor_1/paper14/results/p6d_final_observable_adaptive_v3/confirmation
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6d_final_observable_adaptive_v4_posthoc.py \
  --output Blackdoor_1/paper14/results/p6d_final_observable_adaptive_v4/confirmation
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/build_experimental_coverage_receipt.py --verify
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6d_final_observable_adaptive_v5.py
```

The v1/v2 incident verifiers must return `PASS` with no test access or efficacy
artifacts. The v3 self-check must pass its full-path synthetic shadow. The
stage-aware formal verifier must return `PASS_STAGE_AWARE_FORENSIC_REPLAY`,
five numerical groups with zero score error, and
`INSUFFICIENT_ELIGIBLE_VICTIMS` with count four. These are read-only commands;
never rerun or replace formal seeds 116--123.

The v4 compatibility replay must return a valid ten-victim bundle with zero
score error. Its frozen decision is `INSUFFICIENT_ATTACK_VIABILITY`: seven of
ten attacks meet the utility gate, below the required eight. The coverage
receipt must verify 210 condition--seed assessments and 3,333,660 nested
score/flag records without treating them as independent replicates.

The v5 verifier must return `VALID_DEVELOPMENT_STOP`, selected configuration
`hard60`, attack viability 3/3, viable-and-evasive count 1/3, and
`confirmation_seeds_consumed: false`. The bundle supports integrity and
decision replay, not full numerical score replay because the development stop
did not archive the complete observable matrix. Do not run or repurpose seeds
137--146, and do not use seeds 134--136 for another configuration search.

## QML exact/shot/noise bridge

Replay the completed three-seed primary experiment:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6c_qml_noise_bridge.py \
  --output Blackdoor_1/paper14/results/p6c_qml_noise_bridge_v1
```

Expected output: `status: full_run_valid`, 30 metric rows and 2,592 score
rows.  The verifier distinguishes integrity from performance: the primary run
is valid although the frozen nominal-noise pooled-FPR performance gate fails.

The 2048-shot run is a prespecified diagnostic only and is permitted only after
that primary failure:

```bash
MPLCONFIGDIR=<TMP>/p6c-mpl-cache \
  <USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/run_p6c_qml_noise_bridge.py --diagnostic-2048
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/verify_p6c_qml_noise_bridge.py \
  --output Blackdoor_1/paper14/results/p6c_qml_noise_bridge_v1/diagnostic_2048
```

Do not reinterpret the diagnostic as a second primary experiment or use it to
retune the frozen detector.  The archived diagnostic should verify as
`diagnostic_valid` with 864 score rows and six metric rows; its backdoor-victim
mean AUC is 0.9979 while pooled TPR/FPR remains 1.000/0.713.

## Regenerate paper figures

```bash
MPLCONFIGDIR=<TMP>/paper14-mpl-cache \
  <USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/make_top_journal_figures.py
```

The command writes five PNG/PDF pairs under `results/top_journal_program_v1/figures/`.

Generate the auditability/multiclass Figure 6:

```bash
MPLCONFIGDIR=<TMP>/paper14-mpl-cache \
  <USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/make_auditability_multiclass_figure.py
```

Generate the fresh-query calibration Figure 7:

```bash
MPLCONFIGDIR=<TMP>/paper14-mpl-cache \
  <USER_HOME>/miniforge3/envs/quantum_clean/bin/python \
  Blackdoor_1/paper14/src/make_fresh_query_calibration_figure.py
```

## Regression tests

```bash
MPLCONFIGDIR=<TMP>/paper14-mpl-cache \
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python -m pytest -q \
  Blackdoor_1/paper14/tests
```

The last complete pre-P6-F suite contained 171 tests and passed 171/171 on the
recorded host.  Since then the formal P6-F and TIFS main/supplement surfaces
have been added; their focused replay/build tests pass, while four
release/public-profile inventory tests correctly report the deliberately stale
preterminal manifests.  Do not quote 171/171 as the current suite.  After the
terminal PathMNIST result is integrated, rebuild the profiles and rerun this
full command before refreshing the final-bundle receipt.

## Re-run decisive experiments

The following are intentionally separate because they retrain models and cost substantially more than the replay:

```bash
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_whitebox_adaptive_confirmation.py --data-root <TMP>/blood-ca-data
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_residual_reupload_confirmation.py --data-root <TMP>/blood-ca-data
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_strong_baselines.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_mechanism_ablations.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_covariance_free_confirmation.py --data-root <TMP>/blood-ca-data
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_boundary_study.py
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_post_sanitization.py --data-root <TMP>/blood-ca-data
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_official_backdoorbench_pilot.py --stage confirmation --output Blackdoor_1/paper14/results/bloodmnist_official_backdoorbench_confirmation_v1
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_multiclass_backdoorbench_pilot.py --stage development
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_multiclass_backdoorbench_pilot.py --stage confirmation
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_multiclass_agpd_adapter.py --stage development
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_multiclass_agpd_adapter.py --stage confirmation
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_fresh_query_calibration.py --stage development
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_bloodmnist_fresh_query_calibration.py --stage confirmation
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_pathmnist_external_validation.py --stage development
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_pathmnist_external_validation_v2.py --stage development
<USER_HOME>/miniforge3/envs/quantum_clean/bin/python Blackdoor_1/paper14/src/run_pathmnist_warmstart_pilot.py
```

Before rerunning an experiment, read its protocol JSON under this directory. Existing outputs are evidence and should not be overwritten casually; use a new output directory for any altered setting.

## Interpretation constraints

- Boundary `poison_fraction` is triggered-candidate prevalence under fixed victims, not a victim-training poisoning ratio.
- The original SPECTRE/QUE and SCAn-style comparison is clean-room; the addendum directly calls pinned upstream feature primitives but is still not a bit-identical end-to-end BackdoorBench run.
- Official SPECTRE receives oracle contamination budgets; SCAn is structurally degenerate on the binary benchmark.
- The eight-class confirmation shows that SCAn is highly effective once its across-class anomaly statistic is identifiable; the binary 0/0 result must not be generalized.
- The AGPD result is a declared two-layer feature adapter with pre-frozen compatibility repairs. Its low TPR is an applicability boundary, not evidence that native AGPD is generally ineffective.
- The paired drift view is a clean-checkpoint oracle counterfactual. The frozen Phase-A probe rejects a general drift-over-absolute advantage.
- Training-member candidate scores are not automatically exchangeable with validation calibration scores; call current outputs clean-calibrated rank p-values unless a shift/stability bound is supplied.
- Fresh test queries reduce FPR for suspect hidden64 but increase it for paired drift; membership alone does not determine calibration shift, and official validation/test split exchangeability is not proved.
- The covariance-free adaptive block has only two eligible victims and is inconclusive.
- The lightweight PathMNIST routes remain victim-feasibility boundaries: one of four from-scratch victims and zero of two warm-start victims passed the frozen CA/ASR gate. The disjoint official-weight 103--107 block is conditional detector confirmation only for its frozen dataset, ResNet-18 architecture, patch attack, reference budget, and calibration law; it does not rehabilitate the failed lightweight rows or establish general transfer.
- The PathMNIST warm-start run verifies coordinate alignment, but its ASR is only 3.41%/3.04%; it therefore cannot test the conditional drift-advantage theorem under an eligible attack.
- QML-Z is a negative ablation; XZ16 was later frozen and independently tested.

<!-- PATHMNIST_TERMINAL_BLOCK_V2_START -->
<!-- PATHMNIST_TERMINAL_ROUTE=R3_ELIGIBLE_PRIMARY_GATE_PASS -->
<!-- PATHMNIST_CLAIM_CAPSULE_SHA256=1f86aa9026754f200837109096286bdf4bf316a00313af814a51863a038eb894 -->
## Terminal PathMNIST route: R3

- Decision: `SUPPORTED_CONDITIONALLY`; all 5/5 frozen official-weight ResNet-18 victim pairs are eligible and independently replayed.
- Frozen primary endpoint: 99.79% pooled TPR, 0.54% all-clean FPR, and 3.56% target-clean FPR.
- Scope ceiling: this is conditional confirmation for this dataset, architecture, attack, reference budget, and calibration law only; it does not establish universal or cross-architecture detection.
- Authoritative rows and intervals: `pathmnist_terminal_claim_capsule_v2_2026-10-04.json`; independent capsule verification: PASS.
<!-- PATHMNIST_TERMINAL_BLOCK_V2_END -->
