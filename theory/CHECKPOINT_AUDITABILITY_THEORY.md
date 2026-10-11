# Checkpoint auditability: formal model, guarantees, and impossibility boundaries

## 1. Scope

This document formalizes a conditional audit problem.  It does **not** assert
that every effective backdoor must be visible in a chosen representation.
Instead, it separates:

1. information present in the defender's observable channel;
2. finite-sample ability to estimate a poison direction;
3. calibration validity for the queried clean population; and
4. operational availability and alignment of two checkpoints.

The current proofs are architecture agnostic.  Classical activations and QML
measurement vectors differ only through the observable map.

The results below are modular applications of standard statistical tools.  No
standalone theorem novelty is claimed.  In particular, this document does not
provide an end-to-end guarantee for the implemented cross-fitted, reference-
estimated, multi-view pipeline.

## 2. Audit experiment and threat model

Let \(\theta_0\) be a trusted checkpoint and \(\theta_1\) a suspect checkpoint.
For a fixed measurable representation \(h_\theta:\mathcal X\to\mathbb R^d\),
define the paired audit channel

\[
D_h(x)=h_{\theta_1}(x)-h_{\theta_0}(x).
\]

The strongest version of the model assumes that \(\theta_1\) is warm-started
from \(\theta_0\), so representation coordinates are aligned.  If the models
are trained independently, subtraction is an exploratory comparison unless an
alignment map is estimated without using poison labels.

The defender has:

- a trusted reference sample \(R=(R_1,\ldots,R_r)\);
- a disjoint clean calibration sample \(C=(C_1,\ldots,C_c)\);
- a candidate pool \(B=(B_1,\ldots,B_n)\);
- query access to the declared observables at \(\theta_0,\theta_1\).

Candidate drift follows the mixture

\[
P_\pi^D=(1-\pi)P_0^D+\pi P_1^D,
\]

where \(P_0^D\) and \(P_1^D\) are the clean and poison drift laws.  In the
baseline threat model the attacker controls the poisoned data and suspect
training, knows the detector, but cannot modify \(\theta_0,R,C\).  Extensions
allow an \(\varepsilon\)-contaminated trusted sample.

The estimand is a sample's audit status with respect to the declared channel;
it is not a causal claim that the sample participated in training unless the
data provenance design separately establishes membership.

For a candidate (B_i), let (Z_i\in\{0,1\}) denote its clean/poison status and
let (O_i=h(\theta_0,\theta_1,B_i)) be its declared pointwise observable.  A
pointwise rule has the form \(\varphi_i(O_i)\).  The implemented detector is
batch adaptive: it also uses the other candidate observables, reference and
calibration samples, fold assignments, fitted directions and other algorithmic
randomness.  Let

\[
\mathsf O_i=(O_i,\mathsf A_i)
\]

denote the **full per-decision observation**, where \(\mathsf A_i\) contains
every auxiliary random object supplied to the decision rule.  Write

\[
P_{z,i}^{\mathrm{full}}=\mathcal L(\mathsf O_i\mid Z_i=z),\qquad z\in\{0,1\}.
\]

A batch-adaptive rule has the form \(\psi_i(\mathsf O_i)\).  Its information
bound is governed by the two full conditional laws above, not merely by the
marginal laws of \(O_i\).  All conditioning events must have positive
probability (or be interpreted via regular conditional distributions).

## 3. Three distinct auditability quantities

### 3.1 Intrinsic observable auditability

Throughout, total variation uses the convention

\[
\operatorname{TV}(P,Q)=\sup_A|P(A)-Q(A)|
=\sup_{0\leq f\leq1}|\mathbb E_Pf-\mathbb E_Qf|.
\]

The pointwise marginal quantity is

\[
\mathcal A_h^{\mathrm{pt}}=\operatorname{TV}(P_1^D,P_0^D),
\]

whereas candidate (i)'s full per-decision quantity is

\[
\mathcal A_{h,i}^{\mathrm{full}}
=\operatorname{TV}(P_{1,i}^{\mathrm{full}},P_{0,i}^{\mathrm{full}}).
\]

Both quantities are rule independent, but they govern different rule classes.
The pointwise quantity bounds only rules based on \(O_i\); the full quantity
bounds rules based on \(\mathsf O_i\).  Projection/data processing gives
\(\mathcal A_h^{\mathrm{pt}}\leq\mathcal A_{h,i}^{\mathrm{full}}\), so marginal
indistinguishability does not by itself imply impossibility for a batch-adaptive
rule.  If the chosen QML measurements discard the backdoor-bearing subspace,
\(\mathcal A_h^{\mathrm{pt}}\) can be zero even when ASR is high.

### 3.2 First-order coherence

When \(P_0^D\) has covariance \(\Sigma_0\) and the two means are
\(\mu_0,\mu_1\), define

\[
\mathcal C_h^2=(\mu_1-\mu_0)^\top
\Sigma_0^{-1}(\mu_1-\mu_0).
\]

For singular/high-dimensional problems use a declared regularized covariance
\(\Sigma_{0,\lambda}=\Sigma_0+\lambda I\) and denote the resulting quantity by
\(\mathcal C_{h,\lambda}\).  This is a first-moment quantity: a symmetric
multi-direction poison distribution may have \(\mathcal C_h=0\) while
\(\mathcal A_h>0\).

### 3.3 One-sided sufficient direction-estimation diagnostic

Proposition 5 conditions on the realized poison indicators before controlling
the centered observation noise.  In oracle-whitened coordinates, suppressing
only constants and logarithmic factors, its sufficient signal-to-noise
comparison is

\[
\overline Z\,\mathcal C_h
\gtrsim
\|b_w\|_2+K_E\sqrt{d/m}+K_R\sqrt{d/r}.
\]

The key correction is that the noise terms do not contain
\(\|\delta_w\|_2\).  Conditional on any realized \(\overline Z>0\), angular
alignment tends to one as signal grows.  Hence there is no
dimension-dependent, signal-independent prevalence floor.  The Bernoulli
version still pays a dimension-free lower-tail probability for seeing too few
poisons; a fixed-cardinality cohort or a randomly formed half should instead
use its known count or a sampling-without-replacement bound.

The displayed comparison is a **heuristic sufficient-signal diagnostic**, not
a power, two-sided equivalence, necessary boundary, ASR surrogate, or
end-to-end guarantee.  Effective-rank sharpening requires an additional
covariance-sensitive concentration theorem and is not implied by Proposition
5.

## 4. Proven propositions

The tools used here are standard: the variational characterization and data-
processing inequality for total variation and classical testing theory
(Lehmann and Romano, 2005); exchangeable-rank calibration (Vovk, Gammerman and
Shafer, 2005); Gaussian projection power and Fisher linear discrimination
(Anderson, 2003); sub-Gaussian concentration via an epsilon net (Vershynin,
2018); Hoeffding's lemma; Loewner-order inversion (Bhatia, 2007); and
Cauchy--Schwarz.  The synchronized English and Chinese manuscript
bibliographies contain complete entries, and Appendix A states all eight
propositions and their proof boundaries.  No standalone novelty is claimed for
these standard tools.

### Proposition 1 — Observable-channel impossibility

Let \(0\leq\alpha\leq1\).  For any randomized decision rule
\(\varphi(O)\in[0,1]\) based only on an observable \(O\), if

\[
\mathbb E_{P_0^O}\varphi\leq\alpha,
\]

then

\[
\mathbb E_{P_1^O}\varphi
\leq \alpha+\operatorname{TV}(P_1^O,P_0^O).
\]

**Proof.**  The variational definition of total variation gives
\(|\mathbb E_{P_1}\varphi-\mathbb E_{P_0}\varphi|\leq\operatorname{TV}(P_1,P_0)\)
for every measurable \(0\leq\varphi\leq1\).  Add the FPR constraint.  \(\square\)

Applied to (O_i), this bounds only pointwise rules by
\(\mathcal A_h^{\mathrm{pt}}\).  Applied to \(\mathsf O_i\), it bounds
batch-adaptive rules by \(\mathcal A_{h,i}^{\mathrm{full}}\).  Consequently,
if an attack makes the **full per-decision law** equal under \(Z_i=0\) and
\(Z_i=1\), every rule restricted to that full observation has
\(\mathrm{TPR}=\mathrm{FPR}\).  Equality of the pointwise marginal laws alone
does not imply this conclusion for a batch-adaptive rule.  High ASR alone
supplies no lower bound on either total-variation quantity.

### Proposition 2 — Contraction under shots and noise

Let \(K\) be one common, hypothesis-independent Markov kernel applied under
both clean and poison laws, representing finite-shot measurement, device noise
or randomized post-processing.  Then

\[
\operatorname{TV}(P_1^D K,P_0^D K)
\leq\operatorname{TV}(P_1^D,P_0^D).
\]

**Proof.**  For any output event, integrate its kernel probability against the
two input laws and apply Proposition 1's variational inequality.  Taking the
supremum over output events proves the result.  \(\square\)

Such a common kernel can regularize a particular finite-sample algorithm, but
it cannot add population-level distinguishing information to the declared
channel.  Label-dependent acquisition, adaptive shot allocation with different
policies across hypotheses, or uncontrolled device drift is not represented by
one common kernel and is outside this proposition.

### Proposition 3 — Finite-sample rank calibration

Let \(c\geq1\), \(0\leq\alpha\leq1\), and let
\(S_1,\ldots,S_c,S_{c+1}\) be exchangeable clean scores, with randomized tie
breaking or the conservative p-value

\[
p=\frac{1+\sum_{j=1}^c\mathbf 1\{S_j\geq S_{c+1}\}}{c+1}.
\]

Then

\[
\Pr(p\leq\alpha)
\leq\frac{\lfloor\alpha(c+1)\rfloor}{c+1}
\leq\alpha.
\]

**Proof.**  Under exchangeability, the descending rank of the query is uniform
on \(\{1,\ldots,c+1\}\) without ties; conservative ties can only increase
\(p\).  Count the ranks satisfying the rejection inequality.  \(\square\)

This statement does not automatically apply when the candidate was used to
train \(\theta_1\) but calibration samples were not.  A valid transfer statement
must control the joint rank experiment.  Let \(\mathscr F\) contain the
reference, checkpoint, direction-estimation data, fitted transforms and all
other objects conditioned upon.  For each realized \(\mathscr F\), let
\(\mathbb P_{\mathscr F}^{\mathrm{joint}}\) be an exchangeable benchmark law for
\((S_{c+1},S_1,\ldots,S_c)\), and let
\(\mathbb Q_{\mathscr F}^{\mathrm{joint}}\) be its actual conditional law for the
queried clean population.  Proposition 1 applied to the rank-rejection event
gives, almost surely in \(\mathscr F\),

\[
\Pr_{\mathbb Q_{\mathscr F}^{\mathrm{joint}}}(p\leq\alpha)
\leq\frac{\lfloor\alpha(c+1)\rfloor}{c+1}
+\operatorname{TV}\!\left(
\mathbb Q_{\mathscr F}^{\mathrm{joint}},
\mathbb P_{\mathscr F}^{\mathrm{joint}}
\right).
\]

Integrating gives the same bound with the expectation of the conditional joint
TV term.  A useful but stronger product-law corollary is available if,
conditional on \(\mathscr F\), the calibration scores are i.i.d. from
\(P_{\mathscr F}\), the query is independent of them with law
\(Q_{\mathscr F}\), and the benchmark query also has law \(P_{\mathscr F}\).
Then

\[
\operatorname{TV}\!\left(
Q_{\mathscr F}\otimes P_{\mathscr F}^{c},
P_{\mathscr F}^{c+1}
\right)
=\operatorname{TV}(Q_{\mathscr F},P_{\mathscr F}),
\]

so the unconditional excess is bounded by
\(\mathbb E[\operatorname{TV}(Q_{\mathscr F},P_{\mathscr F})]\).  Without this
conditional product structure, marginal score-law TV is not a valid substitute
for the joint TV term.

Bonferroni fusion of \(k\) valid marginal p-values at levels
\(\alpha_1+\cdots+\alpha_k\leq\alpha\) controls family-wise FPR without
requiring independence.

### Proposition 4 — Gaussian conditional power

Let \(0<\alpha<1\).  Suppose a fixed unit direction \(u\) is independent of a
query, \(u^\top\Sigma u>0\), and

\[
X_0\sim N(0,\Sigma),\qquad X_1\sim N(\delta,\Sigma).
\]

The level-\(\alpha\) upper-tail test based on \(u^\top X\) has

\[
\mathrm{TPR}(u)=1-\Phi\!\left(
z_{1-\alpha}-
\frac{u^\top\delta}{\sqrt{u^\top\Sigma u}}
\right).
\]

**Proof.**  Standardize the one-dimensional Gaussian projection under the two
hypotheses.  \(\square\)

At \(\alpha=0.05\), 90% power requires projected signal-to-noise approximately
\(z_{0.95}+z_{0.90}=2.93\).

### Proposition 5 — Oracle/fixed-transform direction-estimation bound

Let \(d,m,r\) be positive integers, \(0\leq\pi\leq1\), and
\(0<\eta<1\).  Fix a deterministic linear transform before observing the
candidate and reference samples; all variables below are in these transformed
coordinates.  The result also holds conditionally on a transform estimated from
data independent of both samples, provided the following assumptions hold
conditionally.  It does **not** cover a whitening transform estimated from the
same reference observations used in \(\widehat v\).

For every unit vector \(u\), call a centered random vector \(X\)
\(K\)-sub-Gaussian when

\[
\mathbb E\exp(tu^\top X)\leq\exp(K^2t^2/2),\qquad t\in\mathbb R.
\]

Let \(X_1,\ldots,X_r\) be i.i.d. mean-zero, \(K_R\)-sub-Gaussian clean
reference observations.  Independently, let candidate labels
\(Z_1,\ldots,Z_m\) be i.i.d. Bernoulli\((\pi)\), and write

\[
Y_i=b_w+Z_i\delta_w+E_{i,Z_i},
\]

where \(\delta_w\in\mathbb R^d\) is the transformed poison displacement and,
conditional on \(Z_i=z\), \(E_{i,z}\) is centered and
\(K_z\)-sub-Gaussian; \(b_w\) is the transformed clean-candidate/reference
mean shift.  Assume independence across candidate observations and from the
reference observations.  Define

\[
\widehat v=\frac1m\sum_{i=1}^mY_i-\frac1r\sum_{j=1}^rX_j,
\qquad
\overline Z=\frac1m\sum_{i=1}^m Z_i,
\qquad K_E=\max(K_0,K_1),
\qquad
L_\eta=d\log 5+\log(6/\eta),
\]

and

\[
h_\eta=\sqrt{\frac{\log(3/\eta)}{2m}},
\qquad
q_\eta=2\sqrt 2\left(
K_E\sqrt{\frac{L_\eta}{m}}+
K_R\sqrt{\frac{L_\eta}{r}}
\right).
\]

Then, with probability at least \(1-\eta\),

\[
\overline Z\geq\pi-h_\eta,
\qquad
\left\|\widehat v-(b_w+\overline Z\delta_w)\right\|_2\leq q_\eta.
\]

Let \(a_\eta=(\pi-h_\eta)_+\|\delta_w\|_2\) and
\(c_\eta=\|b_w\|_2+q_\eta\).  If \(\delta_w\ne0\) and
\(c_\eta<a_\eta\), then

\[
\cos(\widehat v,\delta_w)
\geq\frac{a_\eta-c_\eta}{a_\eta+c_\eta}.
\]

**Proof.**  Condition on the complete indicator vector
\(Z=(Z_1,\ldots,Z_m)\).  The candidate fluctuation is then only
\(m^{-1}\sum_iE_{i,Z_i}\), whose directional sub-Gaussian proxy is at most
\(K_E/\sqrt m\); it contains no \(\|\delta_w\|_2\) term.  Let
\(\mathcal N\) be a one-half net of the unit sphere with
\(|\mathcal N|\leq5^d\).  The standard net inequality, scalar Chernoff bounds,
and a union bound give

\[
\left\|\frac1m\sum_iE_{i,Z_i}\right\|_2
\leq2K_E\sqrt{\frac{2L_\eta}{m}},
\qquad
\left\|\frac1r\sum_jX_j\right\|_2
\leq2K_R\sqrt{\frac{2L_\eta}{r}}
\]

except on an event of probability at most \(2\eta/3\).  Hoeffding's inequality
also gives

\[
\overline Z\geq\pi-h_\eta
\]

except on an event of probability \(\eta/3\).  Hence
\(\widehat v=b_w+\overline Z\delta_w+e\) with
\(\|e\|_2\leq q_\eta\).  Writing
\(u=\delta_w/\|\delta_w\|_2\), triangle and Cauchy--Schwarz inequalities give

\[
u^\top\widehat v\geq\overline Z\|\delta_w\|_2-c_\eta,
\qquad
\|\widehat v\|_2\leq\overline Z\|\delta_w\|_2+c_\eta.
\]

The ratio is increasing in \(\overline Z\|\delta_w\|_2\), so substituting its
lower bound proves the claim.  \(\square\)

If the fixed transform is the population oracle \(\Sigma_0^{-1/2}\), then
\(\|\delta_w\|_2=\mathcal C_h\).  With the identity transform used by CF-CMD-U,
the same statement is Euclidean.  Conditional on any realized
\(\overline Z>0\), the bound tends to one as \(\|\delta_w\|_2\to\infty\) with
\(b_w,K_E,K_R\) fixed; there is no dimension-dependent, signal-independent
prevalence floor.  The Bernoulli version still pays the dimension-free
probability of observing too few poisons.  A fixed-cardinality cohort instead
has a known count, while a randomly formed direction half calls for a
sampling-without-replacement bound.  Effective-rank replacement requires an
additional covariance-sensitive concentration theorem and is not claimed.

### Proposition 6 — Matched reference contamination

If the trusted sample has contamination fraction \(\varepsilon\) from the same
poison law, then

\[
\mathbb E\widehat v=(\pi-\varepsilon)\delta.
\]

**Proof.**  Subtract the reference mixture mean
\(\varepsilon\delta\) from the candidate mixture mean \(\pi\delta\).  \(\square\)

When \(\varepsilon=\pi\), the first-moment direction vanishes; when
\(\varepsilon>\pi\), a one-sided direction reverses.  This statement does not
cover different contaminating distributions, which require a bias term.  It is
only an expectation identity: without further assumptions it supplies no
concentration, angle, threshold, calibration or detection guarantee.

### Proposition 7 — Conditional benefit of a paired difference

Suppose absolute and difference channels have the same poison mean displacement
\(\delta\) and positive-definite clean covariances satisfying

\[
\Sigma_D\preceq\Sigma_A.
\]

Then

\[
\delta^\top\Sigma_D^{-1}\delta
\geq\delta^\top\Sigma_A^{-1}\delta.
\]

**Proof.**  Matrix inversion reverses the Loewner order on positive-definite
matrices.  Pre- and post-multiply by \(\delta\).  \(\square\)

One sufficient generative interpretation is

\[
h_{\theta_0}(x)=s(x)+e_0(x),\qquad
h_{\theta_1}(x)=s(x)+b(x)+e_1(x),
\]

where subtraction cancels shared semantic variability \(s(x)\) without making
\(e_1-e_0\) more variable than the absolute channel.  The covariance order and
equal-displacement premise must be checked empirically.  They are not automatic.
Warm-starting and coordinate alignment imply neither premise and are therefore
not sufficient conditions for this proposition.  The conclusion compares only
population Mahalanobis mean separation, not AUC or arbitrary detector risk.

### Proposition 8 — Oracle Fisher direction dominates the Euclidean mean direction

For known positive-definite \(\Sigma\), the Fisher direction has signal-to-noise

\[
\mathrm{SNR}_{\rm Fisher}=\sqrt{\delta^\top\Sigma^{-1}\delta}
\geq
\frac{\delta^\top\delta}{\sqrt{\delta^\top\Sigma\delta}}
=\mathrm{SNR}_{\rm Euclidean}.
\]

**Proof.**  Apply Cauchy--Schwarz to
\(\Sigma^{-1/2}\delta\) and \(\Sigma^{1/2}\delta\).  \(\square\)

Therefore CF-CMD-U's empirical improvement over whitening cannot be advertised
as a population theorem.  It points instead to covariance estimation error,
distribution mismatch, non-Gaussianity or regularization failure.

## 5. Mapping predictions to experiments

| Phenomenon | Theoretical diagnosis | Required statistic |
|---|---|---|
| low candidate prevalence | possible direction-estimation failure at fixed full experiment | fold poison count, \(\mathcal E_h^{\mathrm{suff}}\), direction cosine |
| representation-matching attack | reduced observable auditability | held-out two-sample/classifier lower bound, mean effect |
| multi-direction attack | \(\mathcal A_h>0\) but \(\mathcal C_h\approx0\) | energy/MMD versus mean shift |
| reference contamination | direction attenuation/reversal and threshold shift | \(\pi-\varepsilon\), clean-score quantiles |
| QML architecture transfer failure | observable projection loses information | matched observable families and full-oracle upper bound |
| shot/device noise | channel contraction plus finite-sample estimator effects | exact-versus-shot paired auditability |
| independent checkpoints | coordinate mismatch can dominate subtraction | warm-start comparison or label-free alignment |
| context-free single-query screening | pointwise channel may rank poisons yet miss the required low-FPR operating point | candidate-independent TPR/FPR and AUC under a fixed trusted state |

## 6. Current empirical status

The frozen Phase-A theory probe used seeds 42--44, middle/output checkpoints,
and Z/XYZ/Full Pauli families.  Suspect absolute features had mean test AUC
0.994, while independently trained checkpoint drift had mean AUC 0.976.
Drift-minus-absolute deltas were negative for both Z checkpoints, approximately
zero for XYZ/Full, and positive only by 0.002 for output-Full.  Thus the current
evidence rejects a general difference advantage and is consistent with the
coordinate-alignment caveat in Proposition 7.

The same probe raised a calibration warning: the maximum train-clean versus
test-clean rank-FPR gap was 0.0556.  With only 36 clean validation and 36 clean
test pairs per seed, this is diagnostic rather than a precise population
estimate, but it prohibits an unqualified distribution-free conformal claim.

A later protocol-frozen geometry diagnostic evaluates the third row of the
prediction table directly on the archived BloodMNIST white-box checkpoints.
Among the five seeds on which both attack variants satisfy the stored victim
gate, the attack labelled multi-direction has a normalized mean shift 0.974
times that of representation matching and a mean resultant length larger by
0.027.  It therefore fails the predeclared first-moment-cancellation gate.
Label-informed five-fold RBF AUC remains 0.9965 with a seed-bootstrap lower
bound of 0.9947 in the multi-direction eligible population.  Thus this archive
supports neither the intended \(\mathcal C_h\approx0\) mechanism nor broad
indistinguishability in the fixed XZ16 channel.  The result is retrospective
and does not convert a label-informed classifier into a deployable detector.

A separate frozen pointwise audit fits three one-class rules using only a fixed
trusted reference/calibration split and then scores every query independently
of every other candidate.  Across 30 archived white-box victims, fused AUC is
0.939--0.965 and member/fresh clean FPR is 0.64%--1.50%, but poison TPR is only
52.67%--60.71%.  No attack mode reaches the predeclared 80% partial gate, so
fresh confirmation is not authorized.  The result demonstrates that useful
ranking information need not yield an acceptable fixed operating point.  It
supports the implemented method's candidate-batch scope, but it neither
estimates \(\mathcal A_h^{\mathrm{pt}}\) exactly nor proves impossibility for all
pointwise rules.

Here the calibration count is 100 and three marginal ranks are Bonferroni
fused.  Since each rank is an integer multiple of \(1/101\), a fused value at
most 0.05 is possible only when a marginal rank is exactly \(1/101\); the next
fused value is \(6/101\).  The rule is therefore more conservative than its
nominal label suggests.  Direct rank calibration of a single scalar fusion
could recover a finer operating grid, but choosing that fusion after observing
the archive would be post-hoc and its validity would still require the
conditional joint-law assumptions of Proposition 3.

## 7. Remaining proof and experiment obligations

- The repaired propositions and proof boundaries are now included as Appendix
  A in both manuscript languages.  Proposition 5 remains intentionally limited
  to an oracle/fixed transform.  Any sharper covariance-sensitive result would
  still require an independent covariance fold or a new random-whitening proof.
- Quantify or upper-bound calibration shift for training members versus fresh
  clean queries; otherwise call the output a clean-calibrated rank score.
- Estimate \(\mathcal C_h\), direction angle and fold poison counts on the
  existing boundary caches before training new victims.
- Any new cancellation attack must optimize and validation-gate realized mean
  shift and mean resultant length before confirmation; an attack name or
  surrogate term is not evidence that \(\mathcal C_h\approx0\).  Energy/MMD or
  a held-out classifier must still test residual higher-order information.
- Any new single-query rule must be specified on evidence disjoint from these
  25,800 observed queries; post-hoc alpha, fusion or scorer tuning cannot revise
  the frozen negative result.
- Treat any warm-start or aligned-checkpoint experiment as a new, separately
  preregistered hypothesis, and measure the equal-displacement and covariance-
  order premises rather than treating alignment alone as a theorem test.

These obligations define a falsifiable diagnostic-and-limits paper even if no
new detector achieves state-of-the-art performance.  An end-to-end theorem for
the actual cross-fitted, reference-estimated, multi-view pipeline remains open.
