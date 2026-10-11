# PathMNIST weight redistribution audit

Audit date: 2026-10-02

## Decision

`TERMS_NOT_CONFIRMED — DEFAULT TO BINARY EXCLUSION`

The official MedMNIST experiments repository is distributed under
Apache-2.0, but the separate Zenodo record that distributes the benchmark
weights does not display a license value in its Rights/License field.  The
repository README links to that record and describes the weights, but it does
not explicitly state that the repository's code license also licenses the
binary weight files.  The CC BY 4.0 license stated for PathMNIST covers the
dataset and must not be silently extended to trained model weights.

This is a conservative release-control decision, not a legal opinion.  It does
not say redistribution is prohibited; it says the present primary-source
record does not prove permission strongly enough for this project to
redistribute the binaries.

## Primary sources checked

- Zenodo record `10.5281/zenodo.7782114`, “Model weights and predictions for
  reproducible benchmarking experiments in MedMNIST v2”:
  <https://zenodo.org/records/7782114>.  The record identifies
  `weights_pathmnist.zip` and its MD5, but the displayed Rights/License field
  has no license value.
- MedMNIST experiments repository and its Apache-2.0 license:
  <https://github.com/MedMNIST/experiments> and
  <https://github.com/MedMNIST/experiments/blob/main/LICENSE>.
- MedMNIST dataset/code repository:
  <https://github.com/MedMNIST/MedMNIST>.  Its README distinguishes the
  Apache-2.0 code license from the CC BY 4.0 dataset license.

## Hash-bound local inventory

The two locally extracted official members and two epoch-5 derived victim
checkpoints remain inside the private evidence workspace for verification.
Their paths, byte sizes and SHA-256 digests are recorded in
`pathmnist_weight_redistribution_audit_receipt_2026-10-02.json`.  The original
archive MD5, member CRC32/MD5/SHA-256 values, pre-metric selection rule and
seed mapping are already frozen in
`pathmnist_official_weights_acquisition_receipt.json`.

## Release rule

Until an explicit weight license or written permission is archived:

1. Do not put the two official `.pth` files in a public code or evidence
   archive.
2. Do not put checkpoints initialized from those weights in a public archive;
   this project makes no legal classification of whether they are derivative
   works and instead applies the same conservative exclusion.
3. A blind-review package also excludes these binaries unless the journal's
   secure supplementary channel and the rights holder both authorize their
   transfer.
4. Publish the DOI, archive/member hashes, extraction/acquisition instructions,
   frozen seed mapping, model definition, protocols, runners, metric rows and
   verifiers.  A reproducer obtains the official files from the original
   record and regenerates the derived checkpoints.
5. Preserve the private hash-bound evidence workspace.  Exclusion from a
   release copy must never delete or rewrite the authoritative checkpoints.

This resolves the release *policy* but not the underlying license ambiguity.
The preterminal all-evidence manifest continues to inventory the binaries with
a HOLD action. The guarded blind-review builder now verifies their source
hashes and filters them from the reviewer profile before staging; staging
itself remains deferred. A later public-profile builder must preserve the same
exclusion.
