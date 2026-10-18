# Why the first coupled federation seed deteriorated

The main observed failure is TD3 heating drift, accelerated by the extra
replay-only adaptation performed after each federation round. There is also a
concrete deployment-gate protection gap. This diagnosis uses one completed
development seed, not independent confirmation or evidence about real reactors.

## Saved Kaggle evidence

Version `359552340`, `uniform-seed-17`, completed four rounds. At 24 target
training shots the lower-is-better costs were 1.38069 local, 3.25170 reset-only,
3.44031 initial-base pull and 3.48064 raw federation. The learned donor content
was only 1.17% worse than pulling the untrained base through the same pipeline.
Most of the gap from local was already present without sharing learned weights.

Raw federation requested mean auxiliary heat of 23.225 MW at the endpoint,
versus 19.241 MW locally and the calibrated fixed command of 12.5 MW. SAC's mean
deposition location and usage fraction stayed near their initial midpoints.
These were tracking failures, not physical-limit events or hard action clipping.

Every repair added 40 critic updates, 20 TD3 actor updates and 40 SAC actor/entropy
updates on existing replay. Four rounds therefore gave TD3 490 critic updates
in repaired arms versus 330 locally, plus head fitting and target/optimizer
reset. The local arm also eventually deteriorated, so removing federation alone
does not make this a reliable heating learner. Coupled donors were already below
the profile competence criterion after their 48 pretraining shots.

The audits found no basic action-unit, Bellman-sign, SAC Jacobian, entropy-update
or finite-horizon terminal bug. SAC's stored next heating proposal is the one
actually used at the next step. Critic gradients reconstructed from checkpoints
favored more auxiliary heat through the early/middle rounds. Historical gradient
logs were not saved; reconstructed probes are explicitly identified as such.

## Controlled local TORAX probe

`scripts/diagnose_coupled_repair.py` forks the archived target learner immediately
before round-2 acceptance. Variants share exactly the same starting actor,
critics, replay and RNG state. Each is evaluated at the original two development
reporting conditions (initial-temperature factors 0.95 and 1.05). These inspected
conditions are not a held-out confirmation set. No new training shots are used.

| Change to the same checkpoint | Mean cost | Change from untouched |
| --- | ---: | ---: |
| Untouched | 0.111649 | — |
| Extra 40 ordinary updates per role only | 0.216302 | +93.7% |
| Reset/head repair/target reset/adaptation, no transfer | 0.256679 | +129.9% |
| Uniform transfer and full repair/adaptation | 0.245312 | +119.7% |
| Heating-side transfer/repair/adaptation only | 0.246411 | +120.7% |
| SAC-side transfer/repair/adaptation only | 0.111192 | −0.4% |
| Uniform transfer/head repair, no adaptation | 0.111649 | 0.0% |
| Uniform transfer/adaptation, keep old target critics | 0.241157 | +116.0% |

The separate local-arm checkpoint at the same round costs 0.077403 and meets
the declared competence criterion. It has a different training history and is
therefore a contextual reference, not another identical-start ablation.

This directly reproduces damage from heating adaptation at this checkpoint.
Extra updates alone cause most of the immediate cost increase. Keeping old
target critics does not remove it. The no-adaptation branch preserves the actor,
so identical immediate performance is expected; it does not prove that the
changed critic will support better future learning. None of the raw-checkpoint
variants achieves competence. Their full traces, source snapshot, frozen
manifest and optimizer diagnostics are in
`results/coldstart_learning_diagnostic/round2_forks/`.

## Corrected gate and live check

Previously the gate compared a repaired federated candidate with a repaired
local fallback. Both could be substantially worse than the untouched controller.
Even rejection installed the adapted fallback, so it could not protect against
damage introduced by the repair/adaptation pipeline itself.

New local code makes `gate_reference="incumbent"` the default. The candidate is
tested against the untouched pre-transfer deployed controller. Rejection retains
its complete actor, critic, target, optimizer, replay and RNG state. Untouched
control optimizer work is correctly recorded as zero. Raw federation and
reset-only research arms are unchanged. Explicit `repaired_control` mode retains
the prior comparison for reproducibility; archived manifests without the field
are interpreted as that legacy mode. The independent analyzer rejects a gate
reference inconsistent with the frozen manifest.

A second live TORAX probe forks the deployed round-2 checkpoint and evaluates
untouched, reset and federated candidates at all three actual gate conditions
(0.97, 1.00, 1.03). Mean costs are 0.111647, 0.256675 and 0.245308 respectively.
The legacy rule accepts the federated candidate, which is about 4.43% better
than the damaged fallback. The corrected rule rejects it on all three conditions:
it is about 119.7% worse than the untouched incumbent. Evidence is in
`results/coldstart_learning_diagnostic/round2_gate/`.

This protects against transfer-side degradation on the finite gate cases. It
does not protect against drift during subsequent ordinary local training, prove
statistical safety, or establish a reliable TD3 baseline. Frozen-partner and
smaller/no offline actor-adaptation studies remain possible next experiments;
no tuning choice has been selected on confirmation data.

Follow-up development work on the same heating drift is recorded in
[`td3_heating_stability.md`](td3_heating_stability.md). Normalizing the TD3
critics' action inputs passed the late-checkpoint and extra-update rule on three
paired local seeds with fixed microwave deposition, then passed on three seeds
with SAC learning alongside it. A separate full local repair check preserved
competence on those three saved coupled checkpoints, although repair worsened
the cost on one seed. These local findings do not retroactively change the
Kaggle results below or test foreign federated critic updates.

## Costs and the current run

The first probe used 18 fresh evaluation episodes/540 action windows and about
140 seconds. The gate probe used 9 episodes/270 windows and about 108 seconds.
Total new physics work: **27 evaluation episodes, 810 windows, zero training
episodes**. Offline replay optimization is separately recorded. The original
906 Kaggle episodes remain historical development cost. These are selected
checkpoint ablations, not additional independent training seeds.

Kaggle version `359624727` uses the immutable transport-fix bundle and still
contains the old learning settings and old repaired-control gate. It can still
show the same deterioration. Its uniform/clipped/local controls remain useful
for diagnosing that protocol, but it is not a validation run for this local fix.
The cloud run has not been changed or replaced mid-experiment. A future study
must freeze the revised gate explicitly and still report raw failed policies.

Validation passed 73 tests: 52 runner/analyzer/reporting/packaging checks, one
additional gate-reference integrity regression, 15 backend tests and five causal
replay tests. Backend tests verify full-state preservation on rejection,
candidate installation on acceptance, explicit legacy behavior and recovery
contract enforcement. Work accounting and check counts are preserved in
`results/coldstart_learning_diagnostic/local_work_accounting.json` and
`validation.json`. The diagnostic script also passes Ruff.
