# Release archive plan

Status: preterminal design. No public archive or submission ZIP is authorized
by this document.

## Three distinct deliverables

1. **Blind reviewer package.** Manuscripts, protocol/claim tables, derived
   evidence, verifier code and the minimum executable dependency closure. It
   must contain no identifying local paths and must preserve negative results.
2. **Public code package.** Project-owned source/tests/configurations after the
   rights holder selects a license. The BackdoorBench subtree remains under its
   own CC BY-NC 4.0 and listed exception terms.
3. **Evidence/model archive.** Derived scores, figures, receipts and selected
   checkpoints under separately confirmed artifact and upstream-weight terms.
   The current PathMNIST weight audit does not find a license value on the
   Zenodo weight record, so official and official-initialized derived binaries
   are excluded by default while DOI/member hashes and regeneration machinery
   remain. The guarded reviewer-profile builder implements this exclusion and
   verifies the source hashes before filtering. Raw MNIST/MedMNIST benchmark
   archives are not redistributed.

Combining these layers prematurely would obscure different licenses and make
anonymous-review sanitization harder to audit.

## Non-destructive packaging rule

Evidence originals, protocol files and their hashes are never edited to hide
paths or identities. A release builder must copy selected files into a fresh
staging directory, replace machine-specific paths only in the derived copy,
record every transformation, then run the verifier and regression suite inside
that staged copy. The original evidence bundle remains the audit source.

## Generated preflight artifacts

- `release_archive_manifest_preterminal_2026-10-02.csv` hashes the union of the
  verifier-defined evidence payload and the complete project-owned executable
  code closure, including the complete non-`.git` BackdoorBench snapshot, and
  assigns a license/action class. This union is required because a verifier's
  explicit evidence dependencies alone did not include every imported module.
- `release_archive_preflight_preterminal_2026-10-02.json` summarizes payload
  size, risk scans and unresolved release blockers. It deliberately reports
  `PRETERMINAL_NOT_RELEASE_READY`.
- The manifest excludes itself and the preflight JSON to avoid recursive hash
  definitions; the final verifier separately authenticates both files.
- `public_code_profile_manifest_preterminal_2026-10-03.csv` and its preflight
  scan the complete project-code closure rather than only verifier dependencies.
  They include frozen protocols and the vendored BackdoorBench snapshot, exclude
  manuscripts/results/model-data binaries and Claude temporary material, and
  remain publication-blocked until the rights holder selects a project license.
- `blind_review_staged_replay_receipt_preterminal_2026-10-03.json` records a
  disposable end-to-end staged-copy verification and reviewer-profile test
  replay. It remains outside the payload hash cycle and does not replace the
  mandatory post-terminal final rebuild.

## Release gates

The release builder stays locked until the PathMNIST terminal route is
integrated, project-code and artifact licenses are selected, upstream weight
redistribution terms are resolved or the frozen binary-exclusion profile is
applied, author-dependent statements are confirmed,
a stable reviewer/public accession exists, and a sanitized clean-environment
replay passes. A preterminal manifest is useful for detecting drift but cannot
be advertised as the final archive.
