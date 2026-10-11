# Reproducibility core

This archive accompanies the manuscript **Poison-Label-Free Recall Certification:
Impossibility, Minimax Cost, and a Rate-Matching Audit Certificate**.

It contains the detector-agnostic certificate implementation, unit tests, synthetic
generators, frozen/preregistered protocol records, compact result tables, independent
verification summaries, and the data needed to redraw the principal statistical
figures. Large victim checkpoints, feature tensors, and complete per-decision tables
are not duplicated in this review-sized archive. Their hashes and verification
receipts are preserved in the included manifests. They should be deposited in the
full public artifact before acceptance or supplied to the editor/reviewers on request.
This compact archive is not sufficient for from-scratch victim training or full-row
replay. Hash receipts establish traceability; they do not replace omitted objects.

## Environment

- Python 3.10 or newer
- Packages in `requirements.txt`

Create an isolated environment, install the requirements, and run from this directory.

## Fast verification

```text
python -m pytest tests -q
P14CERT_OUT=results/synthetic P14CERT_FIG=figs python sims/make_figures.py
python analysis/plot_certified_recall.py 0.025
python analysis/plot_extension_failure_recognition.py
python analysis/plot_trusted_identity_noninterference_v1.py
```

The first command checks the certificate and matched-baseline routines. The remaining
commands redraw the synthetic, certified-recall, cross-setting, and trusted-identity
figures from the bundled tables.

## Reproduction tiers

1. **Certificate-level replay (self-contained).** The bundled library, tests,
   synthetic simulations, matched PAC comparison, and plotting inputs reproduce the
   statistical machinery and its principal synthetic checks.
2. **Review-sized real-victim verification.** Frozen protocol records, compact tables,
   manifests, and independent summaries support aggregate claim checking and provenance
   audits. They do not constitute complete per-row replay.
3. **Full victim reproduction (external full artifact).** End-to-end retraining and
   full-row replay require training code and configurations, checkpoints, feature and
   raw-score handoffs, complete sealed decision tables, and the evaluation-only verifier
   maps. A persistent public location for these objects remains a submission gate.

## Directory map

- `cert/`: population and fixed-data certificate implementation.
- `tests/`: unit tests for the certificate and matched PAC baseline.
- `sims/`: synthetic experiments and figure generator.
- `analysis/`: issuers, verifiers, validation utilities, and plotting code.
- `protocols/`: preregistrations and frozen configuration records.
- `results/`: compact score/decision summaries grouped by evidence block.
- `figs/`: manuscript figures for comparison with regenerated outputs.

## Scope of this archive

The archive supports independent inspection of the statistical certificate and the
reported aggregate evidence. Re-training every victim requires public datasets,
downloaded pretrained weights where applicable, and the full checkpoint release.
Those large objects are intentionally separated from the journal upload bundle.

No API keys, credentials, private sampling keys, absolute user paths, or model binaries
are included. Keys present in `results/identity/` were explicitly revealed after the
corresponding decisions were sealed and are part of the reported audit record.
