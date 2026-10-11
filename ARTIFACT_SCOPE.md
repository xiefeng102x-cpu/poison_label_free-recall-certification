# Artifact scope

This public repository has two verification layers.

1. `certificate_core/` is the review-sized, self-contained layer. It supports
   unit testing of the certificate, synthetic replay, figure regeneration, and
   inspection of frozen protocol and aggregate verification records.
2. The repository-level `src/`, `configs/`, `tests/`, and `theory/` directories
   expose the broader experiment and victim-training code. They require external
   datasets and, for selected PathMNIST runs, separately acquired benchmark
   weights.

Not redistributed here:

- raw datasets;
- official pretrained-weight binaries;
- trained victim checkpoints and resume states;
- large raw feature and score tensors;
- complete sealed per-decision tables.

Their identities and evidence relationships are recorded by hashes and manifests
in the compact artifact. Hash receipts establish provenance; they do not make an
omitted object replayable. A complete archival deposit remains necessary for
from-scratch reproduction of every reported victim and decision.
