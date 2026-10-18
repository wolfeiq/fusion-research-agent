# Local TD3 heating stability investigation

The objective is a raw heating policy that remains competent before starting a
new federation study. The existing Kaggle source and results stay immutable.
Passing a gate or preserving an already good constant command does not establish
successful reinforcement learning.

The previous checkpoint diagnosis reproduced overheating from extra local actor
updates alone. Both critics preferred excessive auxiliary power. Their action
features occupied only 0–0.25 and 0–0.05 in units of 100 MW; this is a plausible
conditioning problem, not a physical command conversion error. A second concern
is optimizing the actor after only 32 correlated replay transitions.

## Frozen first screen

`scripts/stabilize_td3_heating.py` uses the same corrected-density TORAX model,
calibrated beta_N and electron-temperature profile, 30 half-second action windows,
and original reward. First hold microwave deposition at its calibrated radius and
use all of the EC budget. TD3 still learns both auxiliary power and EC budget.
This removes the changing partner from the heating diagnosis.

Four separate settings are compared from scratch on development seed 17:

| Setting | Only change from original |
| --- | --- |
| Original | None |
| Normalized action inputs | Twin critics receive actions in [-1, 1] |
| Critic burn-in | First 300 critic updates do not update the actor |
| Slow actor | Actor learning rate 0.00001 instead of 0.0001 |

Each receives 48 nominal training shots, including two random warm-up shots.
Evaluate raw deterministic policies at 0, 12, 24, 36 and 48 shots, at initial
temperature factors 0.95 and 1.05. Then apply 160 extra replay-only heating
updates and evaluate again. Keep every checkpoint and every failed setting.
No learned policy is replaced by a conventional controller at evaluation.

The original criterion remains: all 30 windows complete and clean, beta_N MAE
at most 0.005, at least 90% of windows within 0.005, and profile RMSE at most
0.02 keV. A candidate must pass every factor at both late checkpoints and after
the update stress test before promotion to a multi-seed screen. Passing initial
evaluation is reported separately from improvement due to learning.

Budget: 192 training, 48 learner evaluation and two fixed-reference shots,
242 total / 7,260 windows, plus 640 replay-only critic updates. The run is bounded
to 30 minutes, with budget checks between episodes. Exact source, configurations,
per-transition physical traces, training diagnostics and checkpoint hashes are
retained before and during execution. These are inspected development conditions,
not a held-out confirmation set.

The first attempt stopped after four evaluation shots, before training, because
an audit compared pickle bytes that include unstable tensor-storage identities.
The corrected audit hashes actual model, optimizer, gradient, replay, counter and
RNG values. The failed attempt is retained under
`results/td3_heating_stability/screen/`; the corrected attempt is
`results/td3_heating_stability/screen_verified/`.

## Implementation boundaries

Critic coordinates and actor burn-in are explicit TD3 configuration fields and
part of the federation schema identity. Historical callers default to physical
coordinates and zero burn-in; old checkpoint critics retain their original
interpretation. Changing coordinates requires fresh critics, not silently
reinterpreting saved weights. Burn-in continues critic and delayed target-network
updates, and work accounting counts only actual actor optimizer calls. The
existing incumbent deployment gate remains separate from this raw-policy study.

Subsequent work depends on the recorded outcome: test a passing candidate with
seeds 17, 29 and 43, then with an evolving SAC partner. If the fixed-partner screen
fails, record that failure and diagnose it instead of launching federation or
relaxing the tracking requirement.

For the normalized-input candidate, the follow-up is fixed before its final and
stress outcomes are known: repeat original versus normalized inputs on seeds 29
and 43, with the same 48-shot budget, checkpoints, temperature factors and 160
extra replay updates. Seed 17's existing matching comparisons are reused, not
counted as a new seed. Each additional seed costs 96 training, 24 learner
evaluation and two reference episodes (122 total). Separate output directories
allow the independent seed jobs to run concurrently on this laptop. If every
normalized-input seed passes the last two checkpoints and stress, freeze an
equally explicit coupled-partner check before running it. Other first-screen
alternatives remain recorded even if normalization is the candidate advanced.

Conditional coupled screen (frozen before the additional seed results): if the
normalized candidate passes the fixed-partner rule on all three seeds, run that
candidate for 48 shots on each seed with SAC learning on alternating shots. Keep
the original SAC settings, initial controller, targets, reward and observation
schema; no controller fallback. Evaluate at the same five checkpoints and at
temperature factors 0.9, 1.0 and 1.1, covering the proposed donor regimes, followed
by the same 160 extra heating updates. This costs 66 learner episodes plus three
fixed-reference episodes per separately executed seed job (207 total across
three jobs). Test the same competence rule and show
initial performance so improvement cannot be confused with a good constant
initializer. A failure keeps the larger federation rerun on hold.

## First completed screen

The corrected seed-17 screen completed all 242 episodes. Normalized action
inputs are the only candidate meeting the declared late-checkpoint and stress
rule. All initial policies were identical and already competent with the fixed
microwave partner, so this shows preservation of control during learning rather
than a cold-start learning advantage.

| Setting | Cost at 48 shots | Profile RMSE, keV | Cost after 160 extra updates | Late checks and stress |
| --- | ---: | ---: | ---: | --- |
| Original | 2.867513 | 0.322825 | 2.354326 | Fail |
| Normalized actions | 0.076574 | 0.005952 | 0.076889 | Pass |
| Critic burn-in | 1.496864 | 0.230732 | 2.868068 | Fail |
| Slow actor | 0.193058 | 0.063202 | 0.236863 | Fail |

Fixed calibrated command: cost 0.077033 and profile RMSE 0.006032 keV. The
normalized candidate's deterministic heating remains near that operating point.
The other settings show late drift; burn-in and a slower actor postpone it.

Production execution now supports explicit
`--heating-critic-coordinates normalized`. The setting is bound to the saved
study, worker configuration, heating federation schema and recovery contract.
Defaults remain historical physical coordinates; no new Kaggle study has been
launched or silently switched to the candidate. Raw behavior is tested before
using the separate incumbent deployment gate.

## Paired three-seed fixed-partner result

Both additional studies completed and passed the independent artifact audit.
Across seeds 17, 29 and 43, original TD3 fails the last-two-checkpoint and stress
rule in **0/3 passes**; normalized inputs pass in **3/3**. Seed 29 normalized
inputs temporarily miss the profile criterion at shot 12 (mean RMSE 0.021424 keV),
then pass at 24, 36, 48 and after stress. That transient miss is retained.

| Seed | Original final cost | Normalized final cost | Normalized final profile RMSE | Normalized stress profile RMSE |
| --- | ---: | ---: | ---: | ---: |
| 17 | 2.867513 | 0.076574 | 0.005952 keV | 0.005483 keV |
| 29 | 0.097636 | 0.077488 | 0.006283 keV | 0.005995 keV |
| 43 | 0.173887 | 0.081647 | 0.015659 keV | 0.017536 keV |

All reported passes additionally require each condition's beta_N error/coverage
and complete clean physical trace. Seed 43 has less profile margin, so these
three development seeds do not establish unrestricted reliability. Their exact
source archives, full traces, raw policies and optimizer diagnostics are in
`screen_verified/`, `seed29/` and `seed43/` under
`results/td3_heating_stability/`. Independent analysis verifies every recorded
cost and criterion; `fixed_partner_comparison.json` and the same-named PNG combine
the three paired results. No failing arm, checkpoint or seed was dropped.

## Coupled SAC screen and frozen repair check

All three coupled seed studies (`coupled_seed17/`, `coupled_seed29/`, and
`coupled_seed43/`) finished their declared 48-shot training budget. Independent
artifact audits verified all 69 episodes and 2,070 action windows per seed,
including exact source and checkpoint hashes and recomputed raw metrics. Each
normalized TD3/SAC pair passed the original competence rule at both 36 and 48
shots and after 160 additional replay-only heating updates. The initial pair
failed the profile criterion, so this screen includes actual learning from an
incompetent starting policy. Intermediate checkpoints were not monotonic:
seed 17 and seed 29 missed at 24 shots, and seed 43 missed at 12 shots. These
are inspected development seeds, not a confirmation or federation comparison.

The next check is frozen before its results are known. For each of these three
saved coupled checkpoints, fork an untouched policy and run the actual
`LearnerPair.repair()` path with no foreign weights: 25 critic-head fitting
steps, hard critic-target copy, critic optimizer reset, and 40 adaptation
steps per role. Evaluate both raw branches on the same three starting
temperatures (factors 0.9, 1.0, and 1.1) and paired evaluation seeds; no gate
substitution and no new training shots. The unchanged branch must reproduce
the saved final-case costs within the predeclared 1e-8 relative/1e-9 absolute
tolerance. Both branches must be shown, including a failed repair. This is six
physics evaluation episodes per seed, 18 total and at most 540 action windows,
with a 600-second wall budget checked between TORAX calls per job. Preserve
the complete source archive, checkpoint hashes, optimizer work, raw traces,
failures and empirical gate decision for later audit. Passing this check would
establish only that local repair is safe on these three development seeds;
actual foreign critic aggregation remains untested.

The first three repair-check process launches failed before any checkpoint was
loaded or output directory created: `python scripts/check_heating_repair.py`
could not import the repository's `scripts` module on this Windows environment.
The same frozen check was relaunched with `python -m scripts.check_heating_repair`;
the three preflight import failures consumed zero physics episodes.

The three repair jobs completed all 18 evaluation episodes (540 action windows)
without training episodes, limit events or invalid physical values. An
independent reader rechecked the frozen input and source hashes, recomputed the
metrics from every raw trace, checked that no evaluation optimizer updates were
recorded, and confirmed that the untouched policy reproduces the saved
checkpoint costs. The runner also checked complete learner-state digests before
and after evaluation. Both raw branches
meet the competence criterion at every tested starting temperature.

| Training seed | Untouched mean cost | Full local repair mean cost | Relative cost change | Repaired profile RMSE |
| --- | ---: | ---: | ---: | ---: |
| 17 | 0.084944 | 0.078195 | -7.94% | 0.008181 keV |
| 29 | 0.078412 | 0.077495 | -1.17% | 0.012316 keV |
| 43 | 0.077417 | 0.079354 | +2.50% | 0.015318 keV |

Seed 43 worsened on both cost and temperature-profile error after repair, even
though it remained competent and within the declared 5% cost gate tolerance.
The diagnostic gate accepted all three; neither raw branch was replaced by a
fallback for this comparison. The result says the specific local repair path
did not destroy competence on these three development seeds. It does not show
that foreign critic weights will help, that an acceptance gate guarantees
safety, or that federation improves cold-start learning. Source archives,
per-case traces and calculations are in `repair_seed17/`, `repair_seed29/`,
`repair_seed43/` and `repair_comparison.json` under
`results/td3_heating_stability/`.

This investigation used 715 new TORAX episodes / 21,450 action windows:
four evaluation episodes in the retained failed screen attempt; 242 in the
first full screen; 122 each on seeds 29 and 43 with fixed microwave deposition;
207 coupled TD3/SAC episodes; and 18 repair evaluations. Of these, 528 were
training episodes. The extra replay-only update stress and head/adaptation
optimizer work are saved separately and did not consume new TORAX episodes.
These are development conditions and costs, not independent confirmation
trials. The Kaggle federation protocol uses fresh normalized-input critics
explicitly, preserves local and reset-only controls and the incumbent gate,
and accounts for donor and gate-evaluation work.

The Kaggle builder now accepts `--heating-critic-coordinates normalized`, binds
the choice in its launch contract, passes it explicitly to the worker runner,
and includes the TD3 coordinate tests in its private source bundle. A local
candidate bundle was built at
`kaggle/build/coldstart_normalized_td3_candidate/`; its source archive SHA-256
is `27b12726ade4e8798024f652868f1bd1c8a70e28d7c60388f9b77f22bb8cab30`.
The contract says `normalized` and `incumbent`, the notebook command passes the
frozen coordinate setting, and the archive hash was checked against the file.
This candidate was uploaded and launched as private Kaggle version 4
(`359644355`), recorded in `results/coldstart_kaggle_launch/attempt4.json`.
Its cloud suite passed 202 tests (one deselected). Kaggle subsequently
reported successful completion: four of four jobs finished, and the
independent analysis verified 3,624 episodes and 108,720 windows. In the two
paired development seeds, raw federation improved final endpoint cost against
local in seed 17 but worsened it in seed 29. Neither raw federation nor local
reached final competence in either seed, and no learned arm achieved sustained
two-checkpoint confirmation. The calibrated fixed comparator was competent.
The clipped arm clipped zero of 48 source-branch updates, so it was effectively
the uniform arm. This run does not establish a federation cold-start advantage;
its complete cloud report is at the private Kaggle version above. The full
output archive has not yet been preserved locally.

The focused builder/TD3/runner test rerun passed 72 tests, including the new
builder contract test; the separate repair diagnostic rerun passed 21 tests.
