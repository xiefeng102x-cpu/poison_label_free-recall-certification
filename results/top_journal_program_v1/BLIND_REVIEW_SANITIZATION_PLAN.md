# Blind-review sanitization plan

The evidence workspace contains immutable protocol and execution receipts with
machine-specific absolute paths. Editing those originals would invalidate
their provenance. Anonymous review therefore uses a derived copy.

`src/build_blind_review_package.py` implements two modes, and
`src/verify_blind_review_package.py` independently checks the staged result:

- Default audit mode verifies every release-manifest size and SHA-256, computes
  deterministic substitutions in memory and refuses residual user-home or
  private-temporary paths. It performs no package writes; an optional
  `--audit-receipt` stores only the deterministic audit summary.
- The reviewer profile excludes PathMNIST `.pt`/`.pth` binaries from the
  official-weight source, warm-start model directory and terminal detector
  directory. It still verifies their source-manifest hashes before excluding
  them and retains all protocols, metrics, receipts and regeneration code.
- Staging mode requires the exact acknowledgement
  `BUILD_PATH_SANITIZED_BLIND_REVIEW_COPY`, refuses an existing destination and
  refuses any destination inside or above the source `paper14` tree. It copies
  binary files byte-for-byte, substitutes only path prefixes in UTF-8 text and
  writes `SANITIZATION_LEDGER.json` with every source/staged hash and rule count.
- The independent verifier requires exact file-set closure, recomputes every
  staged size/hash, confirms every excluded file matches the narrow weight
  policy, rejects unexpected files and rescans readable text for residual
  paths. It does not trust the builder's success message.

The substitutions are presentation-only:

- POSIX or Windows user-home prefixes become `<USER_HOME>`;
- `<TMP>/` becomes `<TMP>/`.

No sample score, model tensor, threshold, protocol rule, claim decision or
scientific result is changed. A disposable preterminal staging rehearsal is
allowed for QA; it is not the submission package and becomes stale after any
manifest edit. Final staging remains locked until the terminal PathMNIST route
is integrated.

The 2026-10-03 preterminal dry run is recorded in
`blind_review_sanitization_audit_preterminal_2026-10-02.json`: all 841
manifest-bound files pass source size/hash verification; the reviewer profile
contains 837 payload files after excluding two official and two derived weight
binaries. Sixty
text files require 94 user-home and 64 private-temporary prefix substitutions,
and the simulated copy has zero residual paths in those two risk classes. The receipt is kept
outside the payload inventory to avoid a circular dependency on the manifest
that it audits.

The first disposable staging rehearsal passed the builder and independent
hash/path verifier but exposed a missing executable-code closure when the full
test suite was collected inside the copy. A second rehearsal exposed a
sibling-tree QNN evaluator dependency. The release-manifest builder now unions
the complete filesystem-derived project-code closure and verifier runtime
evidence with the scientific payload, and paper14 carries a dense-reference-
tested local evaluator. The current 839-file rehearsal then passed independent
closure/hash/path verification and completed the 171-test staged profile with
166 passes and five explicit policy skips: three require the four deliberately
excluded PathMNIST weight binaries, and two are source-workspace-only packaging
audits. This is preterminal QA, not the final post-PathMNIST submission replay.
