# MNIST historical environment recovery audit

Audit date: 2026-10-03

## Decision

`NOT_RECOVERABLE_FROM_RETAINED_EVIDENCE — CLOSE BY RAW-DATA EXCLUSION`

The exact torchvision version used when the historical MNIST-derived inputs
were generated cannot be recovered from the retained project evidence.  The
current replay pin (`torchvision==0.22.1`) must not be back-projected onto the
older generation event.

This provenance limitation does not change the archived scientific endpoint:
the four retained gzip resources match torchvision's official MNIST MD5
constants and have local SHA-256 values, historical logs show the real
torchvision-loading branch rather than the synthetic fallback, and the Phase-A
manifest hash-binds the six direct tensor/checkpoint inputs actually analyzed.
It does prevent a bit-identical claim about regenerating those inputs from an
exact historical software environment.

## Recovery search

The audit searched project-owned Markdown, JSON, text, log, YAML/TOML, Python
and environment files for torchvision, torch, Python, conda/pip and generation
version records.  It inspected the Phase-A manifest/config/verification files,
retained generator source references, the top-level replay requirements and
the current reproducibility snapshot.  No generation-time package lock,
`pip freeze`, conda export, container digest or historical torchvision version
was found.

The Phase-A manifest records Python 3.9.23 for the later analysis execution but
has no torch or torchvision field.  The current environment snapshot records
torchvision 0.22.1 only for present-day replay.  Neither proves the historical
generation version.

## Redistribution boundary

Torchvision's official repository states that its dataset utilities do not
host or distribute the underlying datasets and do not claim that users have a
license to use them; permission remains the user's responsibility:
<https://github.com/pytorch/vision>.

Accordingly, reviewer and public release profiles exclude the raw MNIST gzip
files.  They retain:

- resource filenames plus official MD5 and observed SHA-256 digests;
- the retained loader call and observed real-data log branch;
- the six hash-bound direct Phase-A inputs;
- analysis configuration, result rows, checkpoints, verifiers and the explicit
  non-regeneration limitation.

This is conservative release control, not legal advice and not a finding that
MNIST redistribution is prohibited.  Reopening this item requires a genuine
historical environment record or explicit dataset terms, not inference from a
modern torchvision installation.

