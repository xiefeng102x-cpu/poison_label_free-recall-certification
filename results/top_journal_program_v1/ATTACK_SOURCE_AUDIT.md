# Cross-attack source audit

Date: 2026-09-28

## Reference implementation and provenance timeline

- Project: BackdoorBench, `https://github.com/SCLBD/BackdoorBench`
- Retrieved archive: `https://github.com/SCLBD/BackdoorBench/archive/refs/heads/main.zip`
- Retrieved archive SHA-256: `0db7869cc9ac470942a78b51405942be4323140687f5dbea83ea2820100b24d6`
- Upstream license: CC BY-NC 4.0 with listed third-party exceptions.
- Initial use on 2026-09-28: read-only protocol/reference audit. The original
  BadNets, Blended and SIG transforms in Paper 14 were written as documented
  clean-room implementations rather than copied from BackdoorBench.
- Later official-baseline work: a pinned BackdoorBench checkout is present at
  `external/BackdoorBench` and is loaded by the feature adapters for upstream
  SCAn, SPECTRE and AGPD primitives. It is therefore third-party executable
  source, not merely a bibliography/reference snapshot.
- Pinned checkout commit: `f02e3534645f0ee63d6848653062cd6c0d6c400d`.
- Bundled upstream license: `external/BackdoorBench/LICENSE`, SHA-256
  `c67d7a90ddf78746757ebff6fcde6ef57cf57c6c9321faa5ffd275565f8f816e`.
- Redistribution boundary: the BackdoorBench snapshot remains governed by its
  CC BY-NC 4.0 license and the third-party exceptions reproduced in that
  license file. A future license for Paper 14's project-owned code must not be
  represented as relicensing the vendored snapshot.

The sentence in the 2026-09-28 version of this audit stating that no upstream
source had been copied became false after the official-baseline snapshot was
vendored. This revision preserves the historical clean-room statement only for
the original direct-transform implementations and records the later change.

## Frozen attack mappings

| Evidence-matrix family | Standard reference | Parameter anchor |
|---|---|---|
| `localized_patch` | BadNets | bottom-right white square; poison ratio 0.1 |
| `global_blend` | Blended | fixed trigger image; train/test alpha 0.2 |
| `geometric_or_frequency` | WaNet and SIG | WaNet `s=0.5,k=4,grid_rescale=1`; SIG `delta=40,f=6` in 0--255 units |
| `sample_specific_input_aware` | Input-Aware/SSBA | learned sample-specific generator; do not replace with arbitrary random noise |
| `clean_label` | Label Consistent/SIG | poison target-class samples without changing labels |
| `pca_additive_fixed` | Paper 14 historical attack | mapped seed-42 direction, PCA `L_inf=0.8` |

BloodMNIST adaptations must preserve the semantic attack family while documenting image-size/channel changes. A configuration is not accepted into the primary cross-attack table merely because a trigger transform runs: its victim must independently satisfy CA >=90% and ASR >=80%.

## Implementation rule

BadNets, Blended and SIG may be implemented directly from their published mathematical transforms. WaNet and Input-Aware require their training-time mechanisms, including cross/noise samples or learned trigger/mask generators; simplified static approximations must be labeled diagnostic and cannot stand in for the named attack in the primary table.
