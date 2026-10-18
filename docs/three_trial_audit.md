# Three TORAX development trials: numerical and reporting audit

All three bounded simulations completed. Their results remain development
evidence: all use the same SPARC-like setup, training seed 29 and one federation
round. Seed 29 was previously used in development; these are three configuration
datasets, with no independent-seed confirmation. The initial agent reports each
had defects. The latest report-only repairs pass their factual checks, with
remaining literature-relevance and optional model-annotation limitations. These
repairs do not establish three fully verified end-to-end workflow passes.

## Frozen comparisons and actual cost

Each question and configuration was frozen before launch. Each baseline and
candidate includes local, reset-only, uniform and clipped arms; exactly one
registered setting changes.

| Case | Setting | Baseline | Candidate | Saved comparison |
| --- | --- | --- | --- | --- |
| SAC actor learning | `sac_actor_lr` | 0.0003 | 0.0001 | [dc0df0a8](../examples/evidence/shadow/dc0df0a81413479e979a18895e1fe2fe/comparison.json) |
| SAC critic mixing | `sac_mix` | 0.25 | 0.1 | [fbb83d8e](../examples/evidence/shadow/fbb83d8ee93a424497aeacc0a2633dc3/comparison.json) |
| TD3 critic action coordinates | `heating_critic_coordinates` | `normalized` | `physical` | [12eddceb](../examples/evidence/shadow/12eddcebd2c548d889b41d3f2adace91/comparison.json) |

Each dataset records **56 episodes: 32 training and 24 evaluation, with 1,680
action windows**. Across the three datasets, actual work totals **168 episodes:
96 training, 72 evaluation and 5,040 action windows**. These totals include both
configurations, donor and target training, and initial, gate and report
evaluation. Earlier calibration and replay-only optimizer work remain separate
costs. Report repairs add no simulation episodes.

The [dataset index](../examples/evidence/3demo-datasets.json) records the exact
configuration, accounting and file hashes. EvidenceCatalog checked the
manifest/config/archive/result chain and recomputed the comparison arithmetic.
All 57 source archive members per dataset match their manifests and the
already-public source bytes at commit `2837fac587a87a07a731d8de199ef32db6df7eb6`.
That commit identifies matching source bytes; it is not a recorded launch commit.
The five numerical/source files per dataset are preserved byte for byte; only
`run.json` is projected to remove process metadata. These checks verify evidence
integrity, not independent physical validity.

## What the numerical results say

The following differences are **candidate minus baseline raw endpoint mean cost
within the same arm**. Negative is better; positive is worse. Values are rounded
to eight decimal places; the linked comparisons retain full precision. These
are cost differences, not percentages or shots-to-competence estimates.

| Changed setting | Local | Reset-only | Uniform | Clipped |
| --- | ---: | ---: | ---: | ---: |
| SAC actor learning rate | +0.00408185 | −0.00883187 | −0.00790225 | −0.00790225 |
| SAC critic mixing | 0.00000000 | 0.00000000 | +0.00012846 | +0.00012846 |
| TD3 critic action coordinates | +0.00071352 | −0.02011896 | −0.01970094 | −0.01970094 |

A separate comparison asks whether federation beats local learning **within
the candidate configuration**:

| Candidate configuration | Uniform raw cost minus local raw cost | Interpretation for this seed |
| --- | ---: | --- |
| SAC actor learning rate 0.0001 | +0.00367020 | Uniform is worse than local |
| SAC critic mixing 0.1 | +0.01578276 | Uniform is worse than local |
| TD3 physical critic coordinates | −0.00476015 | Uniform is better than local |

For example, lowering the SAC actor learning rate improves the uniform arm
relative to its own baseline, while that candidate uniform arm still has higher
cost than its candidate local control. A hyperparameter improvement therefore
does not establish a federation advantage. The TD3 within-candidate result is a
single development-seed observation, insufficient for a general cold-start claim.

All four raw arms in both configurations have `competent=false` in every dataset.
All three recorded recommendations are **`keep_baseline`**. No result establishes
faster shots-to-competence. These tables describe raw learner outcomes; deployed
gate/fallback outcomes are separate fields and can differ. Endpoint cost,
beta_N tracking error and temperature-profile error also remain distinct metrics.

## Reporting failures and repairs

| Case | Initial independent audit | Defect retained in the record |
| --- | --- | --- |
| SAC actor learning | Partial, 62/63 checks | Omitted local competence; literature quotation was generic |
| SAC critic mixing | Insufficient, 59/63 checks | Omitted local cost and reset-only competence; retrieved mirror did not supply an eligible primary reference |
| TD3 critic action coordinates | Physics 38/38 checks; report failed | Three writer attempts failed on malformed optional gaps, missing paired endpoint coverage, then selection above the 12-card limit |

The earlier interrupted request `3ed89057e8d34f16a8eb63ff003cb59f` exposed a
seed-29 request versus fixed-seed-17 tool mismatch before any worker started.
Its zero-physics history is retained. The corrected tool validates an explicit
seed before consuming the attempt and records the same seed in the start event,
frozen configuration and worker evidence.

The reporting fixes require a deterministic current-source bundle: status,
changed configuration, actual accounting, all four paired raw costs, all four
competence comparisons and available candidate SAC clipping diagnostics. Valid
current recommendations are bound to that source. Missing evidence cannot be
filled from historical successes. Malformed optional annotations are quarantined
with a trace entry; they cannot become supported facts. Bounded primary-source
recovery addresses early literature completion without an eligible source.

The first reporting-only repair attempt failed at the provider connection and
is preserved. The second attempt produced all 12 required fact cards in each
case, but the SAC critic-mixing report selected exact bibliographic metadata
instead of scientific content. This defect is also preserved.

The excerpt catalogue now excludes identifiable frontmatter, navigation and
bibliography. Scientific text repeated in a BibTeX abstract can be selected only
when it is unique within the retained scientific body and its original locator
matches an exact candidate returned by the citation helper. Publication repeats
these checks. Forged offsets and text repeated within the scientific body remain
invalid; source bytes, hashes and the 25-word limit are unchanged. The final
public app regression run recorded **1,277 passed, two skips and two existing
warnings**. Automated tests do not establish live scientific interpretation.

The third reporting-only attempt reused the captured numerical and web evidence:

| Case | Independent mechanical checks | Live model calls | Manual literature review |
| --- | ---: | ---: | --- |
| SAC actor learning | 37/37 | 3 | Actual SAC actor reward/entropy objective; methodological context, not a learning-rate explanation |
| SAC critic mixing | 37/37 | 2 | HAPFL scientific abstract about RL managing federated learning; adjacent context, not evidence about federating SAC critics |
| TD3 critic action coordinates | 37/37 | 2 | Actual TD3 discussion of actor-critic overestimation bias; does not establish that physical coordinates reduce bias |

All three repaired factual publications contain the exact 12 current-trial cards,
including all four signed raw endpoint differences and competence comparisons.
They preserve the measured inactive candidate SAC clipping diagnostic: zero of
three contributors clipped, with factors all 1.0. Their source-bound proposals
retain the baseline and leave a possible 112-episode diagnostic follow-up
unlaunched. Independent checks confirmed that original research records,
numerical files and captured local sources remained unchanged. This attempt
used seven model calls, **zero new searches, zero extractions and zero physics
episodes**; it did not rerun the complete research workflow.

The selected primary references were the [SAC paper](https://proceedings.mlr.press/v80/haarnoja18b.html),
the [HAPFL preprint](https://arxiv.org/html/2501.16966v1) and the
[TD3 paper](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf).
HAPFL's surrounding text describes PPO selecting client model sizes and training
intensity. It is not direct support for the SAC critic blend experiment; the
captured canonical federated-RL paper was not selected. Exact quotation and a
recognized primary host therefore still do not establish applicability.

Optional model annotations remain unreliable. In the SAC actor and TD3 cases,
the writer incorrectly described the recorded inactive clipping diagnostic as
missing. The TD3 judge explicitly corrected that note; the SAC actor judge
repeated the confusion. All three causal hypotheses were marked
`critic_rejected` and excluded from the authoritative answer. These are scoped
passes for the server-authored factual publications, not an endorsement of
every model note or three clean full-workflow passes.

The original failures and all repair attempts remain in the audit history.
Repairs changed no current experiment and activated no policy. No result here
establishes a reliable cold-start benefit or authorizes an automatic next run.
