# Poison-Label-Free Recall Certification

Public research code and audit artifact for **Poison-Label-Free Recall
Certification: Impossibility, Minimax Cost, and a Rate-Matching Audit
Certificate**.

The project studies a detector-agnostic question: can a screening run certify,
without poison labels, that its recall meets a target at a declared false-positive
budget? The certificate uses a protected uniform draw from a provenance-certified
clean-member registry. It returns `GO` only when a finite-population upper bound on
the missed-poison fraction is below the allowed ceiling; otherwise it returns
`ABSTAIN`.

## What is included

- `certificate_core/`: self-contained certificate implementation, unit tests,
  synthetic experiments, frozen protocol records, compact result tables, and
  plotting inputs for the current manuscript.
- `src/`, `configs/`, `tests/`, and `theory/`: victim-training, attack,
  detector, verifier, protocol, and audit code from the broader experiment
  program.
- `external/BackdoorBench/`: the pinned BackdoorBench source snapshot used by
  selected adapters, with its upstream license and attribution preserved.
- `results/top_journal_program_v1/`: selected reproduction, provenance, and
  third-party notices. Large result payloads are not included.

## Fast certificate verification

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r certificate_core/requirements.txt
python -m pytest certificate_core/tests -q
```

On Windows PowerShell, replace the activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

To regenerate the bundled statistical figures:

```bash
cd certificate_core
P14CERT_OUT=results/synthetic P14CERT_FIG=figs python sims/make_figures.py
python analysis/plot_certified_recall.py 0.025
python analysis/plot_extension_failure_recognition.py
python analysis/plot_trusted_identity_noninterference_v1.py
```

The full historical experiment environment is recorded in
`requirements-replay.txt`. Some training programs require PyTorch, torchvision,
MedMNIST, public datasets, and separately acquired pretrained weights.

## Artifact boundary

This repository contains the project code and the review-sized certificate
artifact. It intentionally excludes datasets, model checkpoints, official
pretrained-weight binaries, raw feature tensors, and complete per-decision score
tables. The certificate-level replay is self-contained. From-scratch victim
training and full-row replay additionally require the external objects listed in
`certificate_core/README.md` and the provenance documents under
`results/top_journal_program_v1/`.

Hash-bound main experiment sources retain their generic `/private/tmp` defaults so
that the published bytes remain consistent with the frozen protocol records.
Machine-identifying user paths were removed from the public copy. The GTSRB
acquisition helper and its local-search configurations are path-sanitized working
copies; the archived receipts remain the authority for that historical run.

The revealed key under `certificate_core/results/identity/` was disclosed only
after its decisions were sealed and is part of the public audit record. No active
credential, private sampling key, dataset binary, or model binary is included.

## Data and third-party software

MedMNIST and other datasets must be acquired from their official sources and used
under their respective terms. The bundled BackdoorBench subtree remains governed
by its own license and listed exceptions. See
`results/top_journal_program_v1/THIRD_PARTY_NOTICES.md` for details.

Project-owned code is released under the MIT License. This project license does
not relicense the bundled BackdoorBench subtree or any dataset, pretrained weight,
or other third-party component.

## Citation

```bibtex
@article{xie2026poisonrecall,
  title  = {Poison-Label-Free Recall Certification: Impossibility, Minimax Cost,
            and a Rate-Matching Audit Certificate},
  author = {Xie, Zhihai and Zhang, Shibin and Yang, Min and Ren, Mian and Li, Chen},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Contact

Zhihai Xie — xiezhihai@sptc.edu.cn
