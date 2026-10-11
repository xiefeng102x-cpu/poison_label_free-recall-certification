# Preregistration: trusted-member identity non-interference V1

## Question

Can an attacker infer the exact trusted-member identities from information
available no later than the post-training score handoff, before the defender's
secret draw is opened?

This is the appropriate identity-inference experiment for the deployed V2
protocol.  Trusted identities are not fixed during victim training.  They are
sampled only after the victim score handoff by an HMAC-SHA256 keyed RNG.  The
attacker is nevertheless granted a strict information advantage: the complete
verified-clean candidate universe, identifiers, all five detector scores and
flags, and the public sampling algorithm.

## Temporal separation

The attack rankings must be sealed before the defender key is supplied to the
verifier.  The attack runner has no key-file argument and must not read target
membership.  A new 32-byte defender key is committed in the freeze manifest.
The already-open V2 outcome is allowed as a fixed source because the new target
identities do not exist until this new keyed experiment and none of the attack
methods is selected from its outcomes.

## Attack family

For every victim, target detector, fold and trusted budget, the attacker ranks
all verified-clean candidates using six frozen methods:

1. a public identifier hash;
2. the target detector's lowest scores;
3. the target detector's highest scores;
4. the largest standardized deviation over all five scores;
5. selection frequency under 64 public shadow keys; and
6. ridge prediction of shadow-key selection frequency from all scores and
   conventional flags.

The attacker corrupts the first 512 ranked candidates.  The verifier then
derives the formal trusted draws from the unopened defender key.  The oracle-key
ranking is evaluated only as a negative control and is excluded from the safety
family.

## Endpoints

The primary budget is `k=2000`, with 1,000 draws per victim/detector/fold.
Secondary budgets `k=120` and `k=500` use 200 draws.  Each draw reports:

- precision among the 512 attacked candidates;
- recall of the hidden trusted set, which is the realized trusted-set
  contamination fraction;
- lift over the exact random-overlap expectation; and
- AUROC for hidden trusted membership.

Inference uses victim-level means.  There are 60 primary safety cells (six
non-oracle attacks by five detectors by two folds), with Bonferroni allocation
of familywise alpha 0.05.  A cell supports non-interference only when its
two-sided Student-t confidence interval lies wholly inside both equivalence
ranges:

- precision lift in `[0.975, 1.025]`; and
- AUROC in `[0.49, 0.51]`.

The experiment-level primary result is `SUPPORTED` only if all 60 cells satisfy
both conditions, the pooled mean trusted-set contamination does not exceed 10%,
and the oracle negative control attains exact recovery behavior.  Any cell whose
confidence interval excludes the equivalence range is `CONTRADICTED`; all other
failures to establish equivalence are `INCONCLUSIVE`.

## Interpretation boundary

This experiment audits identity leakage through the declared ML observables; it
does not claim security after key disclosure, process compromise, RNG failure,
or metadata side channels that directly reveal the draw.  The existing A1
leaked condition remains the oracle assumption-violation control.  The existing
A2 block remains the downstream test for broad poisoning when exact identities
cannot be inferred.
