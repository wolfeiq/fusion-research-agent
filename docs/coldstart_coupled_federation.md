# Cold-start heating and deposition federation

The primary question is whether experience at two established simulated
facilities reduces the local shots a new target controller needs to become
competent. Architecture comparisons are secondary. This supersedes the initially
proposed symmetric facility comparison before any Kaggle launch.

## Experiment boundary

Use live TORAX 1.4.3 rollouts with corrected density configuration, the calibrated
SPARC-like heating/deposition task, and explicit 50% TD3 initialization. Donors
start in cooler/warmer temperature regimes; the new target starts nominally.
These are compatible simulated regimes of one device model, not three validated
reactor geometries, real experimental data, or evidence of transfer to hardware.
All schedules, seeds, budgets, gate conditions and thresholds must be frozen in
the launch manifest before target results are inspected.

Two donors receive a larger pretraining budget. The target begins with zero
training shots. Each coupled target arm starts with identical parameters and
random streams. Actors, replay, critic heads, target networks, optimizer state
and SAC entropy temperature remain local. No pretrained donor actor is copied
to the target. Critic-only sharing cannot change the zero-shot actor by itself;
any target replay used for repair/adaptation counts as target experience.

## Comparison and aggregation

The main target comparisons are local coupled TD3/SAC, reset-and-repair only,
pull toward the original untrained critic, raw uniform federation and raw clipped
federation. The untrained-base control uses the same 25% mixing and local repair
as federation, separating donor learning from a generic regularization effect.
A separately recorded gated
deployment branch tests rejection of harmful updates. Joint SAC controlling all
four normalized actions and calibrated fixed commands provide architecture and
conventional references. A raw federation failure must not be replaced with its
gate fallback in the main result.

Flower runs real client/server rounds on the Kaggle host. The two roles have
separate channels, and each channel keeps its two critic branches separate.
Clients form deltas relative to the registered common trunk, using frozen local
Bellman labels; local heads are fitted before fitting the copied trunk. Equal
weight aggregation uses `base + mean(delta)`. Clipped aggregation applies the
declared per-branch L2 limit before that mean. The 25% global mix and matched
reset/repair/adaptation control isolate the shared information from reset work.

The development budget is two seeds (17 and 29), two donors with 48 pretraining
shots per architecture, and a target with zero pretraining. Four rounds add six
shots per learned arm, giving 24 target training shots. Uniform and clipped
strategies run separately with paired seeds and starting states. The full job
uses 3,624 simulated episodes (up to 108,720 action windows), including all donor,
control, gate and evaluation work. A three-hour runner deadline preserves partial
artifacts on failure. Source pretraining duplicated across strategy jobs is
charged again; it is not free experience.

This bounded amendment uses synchronous rounds and small MLP trunks. The original
script's asynchronous buffers, similarity/source weighting, server momentum,
profile/set encoders, PopArt and safety critics are not silently claimed here.
Movable TORAX Gaussian deposition is not mirror steering or island suppression.

## Primary outcome

Record deterministic performance at zero target shots and each scheduled target
training checkpoint. Competence requires complete clean trajectories, beta_N
MAE at most 0.005, at least 90% of post-action beta_N errors within 0.005, and
electron-profile RMSE at most 0.02 keV in every declared reporting condition.
The profile tolerance is a development precision goal (about 0.5% of the measured
4 keV core temperature), not an operational reactor standard. It is stricter
than merely maintaining roughly the correct pressure with constant half power.
The initial policy is measured: if it already passes, report zero shots and do
not claim a cold-start improvement for that case.

A first pass observed at shot 12 after a failure at shot 6 means a crossing in
`(6, 12]`, not a measured exact crossing at 12. Report checkpoint-grid first pass
and sustained performance separately. Runs that never pass are right-censored,
not assigned a fabricated successful shot count. Report every paired training
seed and target, cost curves and regressions, not just median endpoint cost.

Account separately for donor training, target training, contribution optimization,
repair/adaptation, development gate rollouts, reporting rollouts, calibration and
actual duplicate control work. Repeated deterministic conditions are not
independent statistical replicates. No confirmation set is used.

## Launch condition

Finish the small local initialization tests first. Run protocol tests and a small
live TORAX/Flower integration check before upload. Preserve a private source
bundle, calibrated reference, dependency lock, manifest, per-round traces and
failures. Use free Kaggle CPU capacity with an explicit deadline and partial
artifact preservation. Do not launch duplicate jobs or use paid resources.

Unreliable local tracking makes this a cold-start diagnostic, not a demonstration
of a reliable controller. The research aim is to test a federation advantage;
the results, including ties and negative transfer, determine whether it exists.

## Local execution check

`results/coldstart_federation_live_smoke/` completed both strategies with real
Flower client/server processes and live TORAX: 184 episodes, 5,520 action windows,
zero sampled events or incomplete cases, and 317.61 seconds of runner wall time.
The independent audit verified the frozen source, all attempt records, the exact
shot budget, role-specific updates, proposal hashes, client acknowledgements and
gate decisions. The smoke used one seed, one round and only two donor/target
training shots; it is execution evidence, not a transfer efficacy study.

At the cold target, endpoint costs were 0.119791 local, 0.105330 reset-only,
0.105203 untrained-base pull, 0.105121 raw federation, 0.102483 joint SAC and
0.077025 fixed commands. Most of the local-to-federation difference was also
present in the repair controls. No learned arm reached profile competence.
Clipping never activated, making the two smoke strategies identical. Preserve
these qualifications when interpreting the larger experiment.

Two additional fixed-command shots at the exact main reporting factors 0.95 and
1.05 passed the whole-trajectory thresholds (Te RMSE 0.006145/0.005919 keV;
beta_N MAE 0.0001201/0.0001205). These 60 windows and 58.94 seconds are additional
development costs, preserved in `results/coldstart_fixed_feasibility/`.

The first private upload was
`kaggle/build/coldstart_coupled_verified/fusion_torax_coldstart_federation.ipynb`.
Its exact extracted source passed all 144 nonphysics tests before submission.
The launch record and preflight hashes are in `results/coldstart_kaggle_launch/`.
Do not call the main results complete until Kaggle's saved version and independent
artifact audit both confirm completion.

The first saved Kaggle version, `359550596`, failed during dependency resolution
after 30.8 seconds, before tests or physics. TORAX's EQDSK dependency declares
Click >=8.3.2, while pinned Flower 1.22 declares Click <8.2. The validated local
environment already uses Click 8.1.8 and EQDSK 0.9.0 successfully. EQDSK imports
Click only in its CLI, which this circular-geometry workload does not invoke.
The retry records an explicit Click 8.1.8 override and pins shared dependencies
to the locally tested versions. This is a scoped research-runtime exception,
not a claim that all EQDSK CLI features satisfy their declared requirements.
The override uses [uv's documented dependency override mechanism](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides).
The failed version, exact source, log and screenshot remain preserved; no
scientific configuration or threshold was changed in response to this failure.

The corrected notebook is
`kaggle/build/coldstart_coupled_constrained/fusion_torax_coldstart_federation.ipynb`.
Its two install steps passed Linux/Python 3.12.3 dry resolution; all scientific
source hashes, experiment configuration, budget and calibration reference match
the first upload. [Kaggle saved version 359552340](https://www.kaggle.com/code/maryk88/fusion-torax-cold-start-federation?scriptVersionId=359552340)
was launched privately on CPU. Its cloud environment passed 144 tests (one live
physics test deselected) in 63.93 seconds. The live TORAX/Flower training runner
then started at notebook elapsed time 102.4 seconds with the frozen 3,624-episode
budget. Kaggle subsequently marked the version failed after 1,149.1 seconds.
At notebook elapsed time 1,137.0 seconds, the runner reported that a Flower
component had failed. The saved study state shows that `uniform-seed-17`
completed and `uniform-seed-29` failed about 10.4 seconds after that job began.
In the latter job, the `donor_warm` client exited 1 with gRPC `UNAVAILABLE` and
`Connection refused` to the loopback Flower server. The server log says it
entered round 1 and terminated gracefully; the other two clients were stopped
during runner cleanup. No physics or aggregation exception is visible in those
component logs. The notebook log, screenshot, component/state previews and frozen
source record are preserved in `results/coldstart_kaggle_launch/`.

## Startup failure diagnosis and transport-only retry

The runner started the server and clients concurrently without waiting for a
listener. Flower 1.22's legacy `grpc-bidi` implementation does not use the retry
invoker passed to it: `max_retries=12` and `max_wait_time=120` therefore did not
protect its first `Join` receive from a refused connection. A delayed real
Flower server reproduced the same error after 2.375 seconds despite those
settings. Flower's termination handler also exits zero and logs a graceful
shutdown when the parent stops it; the old server exit code does not establish
that it failed first. The exact ordering of the original failure was not logged,
but the startup race is a confirmed, independently reproducible defect.

The runner now waits for gRPC readiness before launching any training client,
checks server liveness and bounds startup by 60 seconds and the remaining study
deadline. It captures process states before cleanup, retains a lifecycle log and
prints component log tails on failure. It also checks nonzero exits when every
process has already stopped before the polling loop. It does not retry learner
work or alter training, aggregation, calibration, seeds or acceptance rules.

Validation: 22 runner tests passed, including delayed startup, early server exit,
startup timeout, launch ordering, deadline cleanup and already-exited clients.
Two consecutive real Flower jobs completed two rounds each with the production
readiness barrier; every process exited zero and every update was acknowledged.
These transport tests use synthetic arrays and consumed zero TORAX shots. Their
scripts and logs are in `results/coldstart_transport_debug/`. A broad Ruff check
reports nine existing findings plus import ordering in a new test; this change
does not claim a repository-wide lint cleanup.

## Audited partial scientific result

The full saved output of version `359552340` is preserved under
`results/coldstart_kaggle_audit/359552340/`, including the original ZIP, exact
source, manifests, metrics, summaries, checkpoints and failure logs. Its archive
SHA-256 is `fbc004553e71dc6b8795f55d754f43208c7b7357e0894d74ad76a7365afa8c6b`.
Independent analysis verifies `uniform-seed-17`: four rounds, 906 episodes,
27,180 action windows, no incomplete trajectories or recorded physical events.
The complete matrix remains unfinished. The failed seed-29 job has no recorded
episode attempts and its checkpoints are still at initial round zero; unrecorded
work cannot be recovered from those records.

At 24 target training shots, lower-is-better endpoint costs are:

| Arm | Cost | Competence observed on the checkpoint grid |
| --- | ---: | --- |
| Local coupled TD3/SAC | 1.38069 | Pass at 12 shots; lost at 18 and 24 |
| Reset-only | 3.25170 | Never |
| Initial-base pull | 3.44031 | Never |
| Raw uniform federation | 3.48064 | Never |
| Gated deployment | 3.21975 | Never |
| Joint SAC | 0.11644 | Pass at 18 shots; lost at 24 |
| Calibrated fixed commands | 0.07703 | All checkpoints |

Raw federation's endpoint cost is 152.1% above local, 7.04% above reset-only and
1.17% above initial-base pull. The gate accepts rounds 1–3 and rejects round 4;
gated deployment still deteriorates. No learned arm achieves two consecutive
competent checkpoints. The donor coupled policies already miss the profile
threshold after 48 pretraining shots: mean Te-profile RMSE is about 0.0519 keV
versus the 0.02-keV criterion. Their joint-SAC comparators are much closer to the
target (about 0.0142 and 0.0129 keV). Thus these donors are experienced, but not
demonstrably skilled at this task. Beta_N tracking, profile tracking, endpoint
cost and sustained competence must remain separate in any claims.

This one development seed provides negative evidence for the current coupled
setup, not evidence of a cold-start benefit. A fresh frozen matrix with only the
transport fix is the simplest valid recovery: original manifests and checkpoints
remain unchanged, the previous 906 episodes remain additional development cost,
and a repeat of seed 17 is a replication, not an extra independent seed. The new
matrix retains the same 3,624-episode budget and three-hour execution ceiling.
The completed job took about 17 minutes; roughly 70 minutes is an estimate for
the matrix, not a guaranteed finish time.

The transport-only retry is saved as private CPU
[Kaggle version 359624727](https://www.kaggle.com/code/maryk88/fusion-torax-cold-start-federation?scriptVersionId=359624727).
Its bundle is `kaggle/build/coldstart_coupled_startup_fix/`; source SHA-256
`a8349c1661d4500b6974f62d125c42bdbe50e711cfb567c3797708aa1d881214` was verified
in the cloud log. The comparison against the prior source archive confirms only
the runner and runner tests differ; the only changed production definitions are
`run_job` and the new `wait_for_flower_server`. Configuration, calibration,
scientific source and budget are identical. Launch status and validation evidence
are recorded in `results/coldstart_kaggle_launch/attempt3.json` and
`startup_fix_preflight.json`. Kaggle passed 149 tests (one live physics test
deselected) in 67.86 seconds and started the live TORAX/Flower training runner at
notebook elapsed time 104.4 seconds. This is a fresh run; its results are pending.

Subsequent local diagnosis is documented in `docs/coldstart_learning_diagnosis.md`.
Controlled checkpoint forks reproduced heating-adaptation damage and exposed a
gate that could accept a candidate relative to an already damaged fallback.
New local studies default to an untouched-incumbent gate, with the legacy mode
available explicitly. Running Kaggle version `359624727` is immutable and retains
the older gate and learning settings; its results must be labeled accordingly.
