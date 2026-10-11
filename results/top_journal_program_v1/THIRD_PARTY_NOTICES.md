# Third-party code and data notices

Status: factual inventory for review and archive preparation. This is not
legal advice, a project-code license, or a freedom-to-operate opinion.

## BackdoorBench source snapshot

- Upstream project: `https://github.com/SCLBD/BackdoorBench`
- Local path: `external/BackdoorBench`
- Pinned commit: `f02e3534645f0ee63d6848653062cd6c0d6c400d`
- Local license: `external/BackdoorBench/LICENSE`
- License-file SHA-256:
  `c67d7a90ddf78746757ebff6fcde6ef57cf57c6c9321faa5ffd275565f8f816e`
- Declared upstream license: CC BY-NC 4.0, with separately listed third-party
  components under Apache-2.0, BSD and MIT terms.
- Paper 14 use: the adapters load upstream SCAn, SPECTRE and AGPD feature-level
  primitives. Other directly coded trigger transforms are separately described
  in `ATTACK_SOURCE_AUDIT.md`.

Any distributed archive containing this directory must preserve the upstream
license and attribution, indicate local modifications if any, and respect the
non-commercial restriction. The author must make an informed project-code
license choice; a permissive license placed at repository root would not erase
or replace the terms covering this directory.

## MedMNIST software and datasets

- Reproduction dependency: MedMNIST 3.0.2. The installed distribution's
  package license is Apache-2.0.
- BloodMNIST and PathMNIST metadata in MedMNIST 3.0.2 declare the datasets as
  CC BY 4.0. The project does not redistribute their NPZ archives.
- BloodMNIST evidence uses the archived NPZ SHA-256
  `062023e186f537e26b3c21ea3b2614ddfc475e8a14825dfd20663bbb7e37bddc`.
- PathMNIST evidence uses the acquired NPZ SHA-256
  `81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72`;
  source URL, split counts and acquisition checks are recorded in
  `pathmnist_acquisition_protocol_v1.json` and
  `pathmnist_acquisition_receipt.json`.

Dataset attribution must cite MedMNIST and each relevant source dataset as
required by the dataset documentation. Public availability does not remove
attribution duties or create permission for clinical claims.

## MedMNIST benchmark weights

The MedMNIST experiments source repository is Apache-2.0, but the separate
Zenodo record `10.5281/zenodo.7782114` that distributes the benchmark weights
does not display a license value in its Rights/License field. Neither the code
license nor PathMNIST's dataset license is therefore treated as proof that the
weight binaries may be redistributed. The conservative rule is to exclude the
two official ResNet-18 files and checkpoints initialized from them from
reviewer/public packages unless explicit authorization is archived. Preserve
their private evidence hashes and publish the DOI, acquisition/member hashes,
frozen runners and regeneration instructions instead. See
`PATHMNIST_WEIGHT_REDISTRIBUTION_AUDIT_2026-10-02.md`. This is release control,
not legal advice or a finding that redistribution is prohibited.

## MNIST-derived archived inputs

The QML MNIST block consumes previously generated tensors/checkpoints under the
historical `data/Mnist/detail_single_samle` source layout; the raw dataset is
not included in the Paper 14 bundle. The retained generator uses
`torchvision.datasets.MNIST(train=True, download=True)`. All four local gzip
files match torchvision's official MNIST resource MD5 constants and now have
recorded SHA-256 values; the Phase-A manifest separately hash-binds all six
direct checkpoint inputs. See
`mnist_source_provenance_receipt_2026-10-02.json`.

The historical logs did not record the exact generation-time torchvision
version or hash-bind the retained generator source. A repository-wide recovery
audit found no historical package lock and therefore forbids projecting the
current replay pin backward. Torchvision's official dataset disclaimer also
does not grant underlying dataset permission. Raw MNIST files are consequently
excluded from reviewer/public profiles, while resource and direct Phase-A
input hashes remain. See
`MNIST_HISTORICAL_ENVIRONMENT_RECOVERY_AUDIT_2026-10-03.md`. These are
archive-provenance and release boundaries, not uncertainties about the already
hash-bound Phase-A analysis.

## Project-owned code

Paper 14's project-owned code is released under the repository-level MIT
License. This does not relicense the bundled BackdoorBench subtree, datasets,
pretrained weights, or any other third-party component. Their original terms
remain in force.
