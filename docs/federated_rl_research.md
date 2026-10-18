# Federated RL literature: implications for tokamak aggregation

Companion specification: [Aggregation policy v1](aggregation_policy_v1.md). Status: literature review and proposed design; not a new implementation or experimental result. This is a targeted primary-source review, not an exhaustive novelty search.

## Findings that change the design

The quantity to share must be specified before choosing weights. A reactor-specific critic estimates the return of a particular policy under particular transitions, rewards and constraints. Common tensor shapes or dimensionless inputs do not make these functions identical. Shared representations with local value heads are a plausible way to exploit common structure without assuming a universal Q function. Whether the structure exists for our devices remains an empirical question.

| Source | What was studied | Design implication and limit |
|---|---|---|
| **Jin et al., AISTATS 2022:** [Federated Reinforcement Learning with Environment Heterogeneity](https://proceedings.mlr.press/v151/jin22a.html) | QAvg and PAvg learn across differing transition dynamics; results characterize heterogeneity-dependent suboptimality, with an environment-embedding personalization heuristic. | Do not assume naive averaging removes device differences. Include common-policy aggregation and personalized transfer baselines. The paper does not validate our reactor similarity metric. |
| **Zhu and Gong, AAAI 2025:** [Single-Loop Federated Actor-Critic across Heterogeneous Environments](https://ojs.aaai.org/index.php/AAAI/article/download/34469/36624), [searchable author version](https://arxiv.org/html/2412.14555v1) | Coupled actor–critic learning across environments; shared state/action spaces, heterogeneous transitions and rewards. Convergence bounds include heterogeneity effects under the paper's assumptions. | Useful actor–critic federation baseline, not a guarantee for distinct reactor action spaces, our synchronous TD3 pilot, or proposed asynchronous extensions. See problem formulation and SFAC algorithm. |
| **Xiong et al., ICLR 2025:** [On the Linear Speedup of Personalized Federated Reinforcement Learning with Shared Representations](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9d94bf4711fa459812437e5df5978551-Abstract-Conference.html), [author text](https://arxiv.org/html/2411.15014v1) | PFedRL-Rep shares a feature representation and retains personalized weights. The main analysis treats PFedTD-Rep with linear representations; control extensions are evaluated experimentally. | Closest established basis for shared features/local heads. Section 4 is the theory boundary; its speedup result is not a convergence proof for nonlinear twin-Q control. |
| **Wang, Yang and Su, 2026 preprint:** [Collaborative Yet Personalized Policy Training: Single-Timescale Federated Actor-Critic](https://arxiv.org/html/2605.14423v1) | A common linear subspace, local critic heads and personalized actors; finite-time results under structural and sampling assumptions. Experiments use PPO with action-map heterogeneity. | Particularly relevant to our proposed architecture. The theoretical construction and the deep PPO experiment are different levels of evidence. Neither establishes our TD3 implementation, a future SAC variant, or fusion transfer. See joint approximation objective and Appendix J. |
| **Xie and Song:** [FedKL: Tackling Data Heterogeneity in Federated Reinforcement Learning by Penalizing KL Divergence](https://arxiv.org/abs/2204.08125) | Restricts local policy changes in distribution space to manage heterogeneous policy-gradient updates. | Policy drift is a behavior issue, not merely an integer version gap. Do not present an arbitrary policy-age multiplier as an importance-sampling correction. FedKL's shared-policy setting differs from local actors. |
| **Jin, Zhang and Zhang, 2024 preprint:** [Federated Reinforcement Learning with Constraint Heterogeneity](https://arxiv.org/abs/2405.03236) | Federated primal–dual policy optimization with distributed constraint information; FedNPG and FedPPO. | Constraints need explicit costs, budgets and optimization rules. Adding a network named “safety critic” is insufficient. Their common-policy constraint formulation is not a guarantee for personalized reactor controllers. |
| **Nguyen et al., AISTATS 2022:** [Federated Learning with Buffered Asynchronous Aggregation](https://proceedings.mlr.press/v151/nguyen22b.html), [paper](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf) | FedBuff collects client updates before applying server updates. Section 5 uses inverse-square-root version-staleness scaling as a practical improvement. | Appropriate scheduling precedent for an asynchronous extension. It is an FL optimization result, not a theorem for moving RL targets. Compatibility with secure aggregation does not mean our code implements it. |
| **Wang et al., NeurIPS 2020:** [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | FedNova addresses objective distortion from unequal local computation. | Fix local optimization budgets first. Dividing arbitrary Adam deltas by step count is not a faithful implementation of their normalization. |
| **Collins et al., ICML 2021:** [Exploiting Shared Representations for Personalized Federated Learning](https://proceedings.mlr.press/v139/collins21a.html) | FedRep learns common representations with local heads. | Architectural foundation, but supervised FL guarantees do not automatically cover bootstrapped RL losses. |
| **van Hasselt et al., NeurIPS 2016:** [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714) | PopArt adaptively normalizes targets while preserving unnormalized predictions. | Helps with target scale; it does not reconcile different policies, transition kernels or reward meanings. Normalization statistics and output corrections must stay local if heads stay local. |

## Corrections to the earlier aggregation narrative

- Sample weighting is a choice of objective, not inherently arbitrary: it can target a pooled-example objective. Equal device weighting targets a different objective. We choose equal device weighting initially because devices, not replay-buffer insertions, are the units of interest.
- FedAvg can operate over sampled participating clients; it does not require every registered reactor to be online. Synchronous learning from stored data is possible between shots. Asynchrony is an engineering option to evaluate, not a physical necessity.
- A dimensionless distance may be a useful prior, but it is not a proven distance between Bellman operators. Treat similarity as an ablation, not established physics-based optimal weighting.
- Server momentum alone is not equivalent to a published momentum/variance-reduced FRL algorithm. Leave it disabled in the first experiment.
- One global proposal and target-specific similarity mixtures are different algorithms. Specify which is being run; do not call both a single global critic.
- Shared weights may leak information. “No raw trajectories uploaded” is an auditable data-flow statement, not differential privacy or secure aggregation.

## Recommendation

The synchronous shared-representation/local-actor-and-head baseline is now implemented as a one-time TD3 handover with explicit local targets and equal source weights. The newer three-way diagnostic adds the initial-trunk reset control. Establish stable local learning and compare with single-source transfer before testing buffer scheduling, similarity kernels or adaptive mixing. Preserve the existing policy-search results as a separate experiment family. The linked specification records the implemented subset and proposed extensions without claiming any of the cited theorems for our code.

## Post-audit experiment rationale

**Status: development designs, not observed efficacy results.** The matched-client
scenario is now implemented as a separate protocol; the broader regime/fault and
compatibility-weighting ideas remain proposals. The [pilot audit](kaggle_critic_audit.md) found unstable local TCV
learning and a missing reset-only control. Neither can be solved by selecting a
more flattering federation metric. First examine the local-stability and
three-way diagnostic runs; the experiments below should isolate a mechanism and
retain failed conditions. Positive and negative controls are deliberate tests of
the hypothesis, not opportunities to relabel whichever device happens to win.

### What the primary papers support

| Primary evidence | Relevant result | Boundary for this harness |
| --- | --- | --- |
| [Jin et al., QAvg/PAvg, AISTATS](https://proceedings.mlr.press/v151/jin22a.html) | Their common-policy/value aggregation has heterogeneity-dependent suboptimality; environment embeddings provide an empirical personalization route. | Different dynamics can create bias as well as useful additional experience. Dimensionless similarity is not shown to measure that bias. |
| [Xiong et al., PFedRL-Rep, ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9d94bf4711fa459812437e5df5978551-Abstract-Conference.html), [Sections 4–5 and Appendix G](https://arxiv.org/html/2411.15014v1) | The proved speedup concerns TD with linear representations. Experiments extend to DQN control and DDPG-based Hopper with changed dynamics, including adaptation to a new environment. | This motivates testing reusable representations under controlled heterogeneity. It does not prove nonlinear twin-critic TD3 stability or guarantee improvement when a source learner has collapsed. |
| [Zhu and Gong, SFAC, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34469), [Assumptions 1–3 and Proposition 1](https://arxiv.org/html/2412.14555v1) | Actor-critic federation reduces a sampling-noise term, while environment heterogeneity contributes an error term under the stated boundedness/mixing assumptions. | More contributors can reduce variance without removing systematic mismatch. Our finite-horizon, terminating TORAX task has not been shown to satisfy those assumptions. |
| [Wang, Yang and Su, personalized actor-critic preprint, Section 6.2 and Appendix J](https://arxiv.org/html/2605.14423v1) | PPO experiments vary actuator permutations/scales and transfer the learned shared trunk to new tasks. Frozen-trunk adaptation outperformed full fine-tuning in the reported downstream examples. | Their deep experiment shares a trunk used by **both actor and critic**, with local output heads. Our TD3 actor is entirely separate; only critic trunks transfer. Their direct policy-feature benefit is therefore not the same mechanism. The linear-subspace theory is also distinct from either deep implementation. |
| [Zhang et al., FedFomo, ICLR](https://arxiv.org/abs/2012.08565), [authors' implementation](https://github.com/NVlabs/FedFomo) | Personalized model combinations use local validation performance to favor useful peers in supervised federated learning. | This motivates a compatibility-weighting ablation, but replacing supervised validation with bootstrapped TD error is our unproven adaptation. Low TD loss need not imply good control. |

Our falsifiable working hypothesis is that useful source experience can reduce the
target data needed to learn critic features, provided source/target tasks share
structure, the local learner remains stable, and the target retains its own
policy and value heads. This is an inference from the papers, not a conclusion
about our existing run. The hypothesis predicts a larger benefit when target
data are scarce and source coverage is relevant, and weaker or negative transfer
under incompatible dynamics. Test those interactions instead of assuming them.

### Three tractable scenarios

Keep the current constant-transport TORAX model for these mechanism tests. They
do not become realistic turbulence or reactor-operation benchmarks simply by
using more conditions. Use the same actuator deposition semantics, reward,
observation schema, physical horizon and optimizer configuration within each
comparison. All numerical settings below are proposed development settings.

| Scenario | Construction and implementation boundary | Falsifiable prediction and decisive control |
| --- | --- | --- |
| **Matched dynamics, scarce target data** | Use independent simulated clients with the same SPARC-like geometry/task and disjoint trajectories. Two source clients have 40 training shots each; vary the target's pre-handover data at 10 versus 40 shots while keeping the final target budget at 120. This needs scenario/client identifiers separate from device names; the current disjoint-device check must not be bypassed without an explicit new study mode. | If the contribution/repair mechanism transfers useful features, learned-source transfer should improve early held-out control relative to both local and reset-only arms, especially at the smaller target budget. Failure here weakens this implementation before invoking cross-device physics mismatch. This is a homogeneous positive control, **not** evidence of transfer between different reactors. |
| **Controlled campaign shift** | Keep one geometry and actuator map; vary pedestal temperature using the existing `T_ped_keV` configuration override. Start with source factors 0.9 and 1.1 times nominal, target factor 1.0 for interpolation and 1.25 for an extrapolation stress condition. The study runner must expose these overrides and record resolved configurations. Recheck reachable setpoints/error floors for each configuration: the registry's nominal-device bands must not silently certify altered plants. | Useful common thermal structure predicts better transfer within source coverage than beyond it. If extrapolation is equally good, that refutes the proposed coverage interaction, not the whole method. Use a calibrated conventional controller to verify feasibility; retain the identical target-only and reset-only controls. Freeze any revised feasible schedule before comparing learners. |
| **Actuator response mismatch** | With fixed geometry, multiply commanded ECRH power by a declared plant gain; preserve the Gaussian deposition shape. Source gains 0.6 and 1.0, target gains 0.8 and 0.4 separate interpolation from extrapolation. This requires a tested RL-environment gain hook; it is not already provided by the LLM app's thermal tool. Log commanded and delivered power separately. Keep the gain fixed within each learner/run, with the same observability contract in all arms. | Local heads may absorb moderate response mismatch while retaining useful features; severe out-of-range mismatch may cause negative transfer. Test whether compatibility weighting helps specifically in the mismatch condition. Do not claim to detect an unknown heater fault if the true gain is supplied as an observation; do not silently alternate hidden gains between episodes. |

The existing cross-device ITER/DIII-D-to-SPARC/TCV experiment remains the reference
condition. Source-quality screening, if introduced, must be a declared
source-development rule with rejected-source counts; it must not retrospectively
remove a difficult source because the target confirmation result was unfavorable.
Do not spend the full budget on the cross-product of every scenario, algorithm,
target budget and seed. Establish the matched-dynamics control first, then vary
one source of heterogeneity at a time. Both lower and higher target budgets must
contain enough replay for the predeclared batch size and repair procedure.

### Three aggregation strategies with exact meanings

All strategies start from the same registered common base `phi_0`. Each source
creates its disposable, frozen-label representation job with the same head/trunk
step budgets. Both critic branches are present in every contribution. The target
installation is always `0.75 * phi_local + 0.25 * proposal`, followed by identical
head repair and actor-learning opportunities. Actors, replay, targets and local
heads stay local. Keep a single handover while comparing weighting rules; changing
communication frequency at the same time would add another explanation.

1. **Uniform source deltas:** `proposal = phi_0 + mean_i(delta_i)`. This is the
   existing reference. Prediction: it can work in the matched condition, but is
   vulnerable to a poorly matched or disproportionately large source update.
   Equal coefficients do not imply equal vector influence.
2. **Norm-clipped uniform deltas:** for branch `k`, use
   `delta_clipped_i,k = delta_i,k * min(1, C_k / ||delta_i,k||_2)`, treating a zero
   norm as an unchanged zero delta, then average. Set `C_k` from the median
   contribution norm in independent development runs and freeze it for the later
   comparison; record every factor and pre/post norm. This is our engineering
   ablation, not a cited robustness or privacy theorem. Prediction: it helps when
   large harmful deltas dominate; it may harm when the largest update is useful.
   Compare against a direction-preserving, source-delta-norm-matched uniform update
   as a diagnostic. This controls learned source-delta magnitude, not the full
   installed target step, whose reset offset also matters. Record both before
   attributing an improvement to identifying useful contributors.
   Clipping can reduce cancellation and require amplification rather
   than shrinkage; a zero uniform mean cannot match a nonzero clipped mean.
   Record these cases explicitly, as specified in the
   [matched-client plan](matched_client_experiment_plan.md). Such a control is
   not a fourth proposed aggregation policy.
3. **Target-calibration compatibility weights:** reserve two additional target
   episodes before handover for calibration, never for gradient updates or final
   evaluation; charge them to every arm's target-data budget. Freeze the incumbent's
   target networks, smoothing draws and TD labels on these episodes once. For each
   `phi_0 + delta_i` and the reset-only `phi_0`, apply the same target mix and fit
   heads on target training replay, then compute normalized twin-critic squared
   TD error on calibration data. With losses `L_i` and `L_reset`, define
   `s_i = max(0, L_reset - L_i)` and `w_i = s_i / sum_j(s_j)` if the sum is positive.
   Use `proposal = phi_0 + sum_i(w_i * delta_i)`; if no source improves this proxy,
   record an abstention and retain the matched local control. Record raw proposals
   separately from abstentions. Extra peer-scoring optimization must be reported
   and matched by a local-compute control if efficiency is claimed. Prediction:
   weights improve the mismatched-dynamics condition only if calibration TD error
   predicts downstream control. Compare true versus shuffled source-to-weight
   assignment and report that correlation. If it fails, reject this proxy rather
   than retrospectively replacing it with final-return weights.

Compatibility weights must use one combined twin-critic score per source so both
branches retain the same source cohort/weights. Its calibration episodes reduce
the apparent scarcity advantage and must not disappear from the cost accounting.
Bootstrapped labels can be wrong, especially under limited action coverage; this
is why strategy 3 comes after a stable local learner, not as a proposed cure for
the observed TCV collapse. Clipping and compatibility weighting are evaluated
separately initially, not combined into an uninterpretable package.

### Controls, decisions and reporting

Every source-sharing comparison retains **local with matched repair**, **initial
trunk reset with zero source update**, and **learned-source transfer**. Single-source
ablations identify whether averaging adds anything beyond the better source. A
pooled-data reference, if added, uses the same source trajectories and discloses
that raw-data pooling is allowed; better performance than local alone does not
establish superiority to centralized learning.

Use source training, target training, development/calibration and confirmation
identifiers with explicit disjoint membership. Split whole episodes, not adjacent
transitions, for calibration. Fresh random seeds from the same rounded initial
temperature factors are not a new operating-regime split. Preserve raw policies
at every evaluation checkpoint; report raw learner failures even if a separate
retained controller or rollback rule performs well. Continue to report termination
events as simulator limit violations, not real disruptions.

During development, report paired differences in deterministic evaluation cost
at fixed target budgets, tracking error, violation rate, and the fraction of
training seeds that learn a viable controller. Count training shots, calibration/
gate shots, transitions, optimizer steps, source preparation and wall time
separately. A lower endpoint cost is not a measured speedup. Shared source cost
may be amortized over several new targets only if that target count and the
amortization rule are stated; also show the unamortized total.

Before confirmation, freeze an absolute competence definition, validation cadence,
required persistence across checkpoints, failure/censoring rule and the effect
worth detecting. Select the independent paired-seed count from development
variability and that effect, not from the number of evaluation rollouts. Lock all
source cohorts, gains/pedestals, clip thresholds, weight rules and code before
opening confirmation outcomes. If the local baseline remains unstable, retain
the result as a development failure and investigate learner/action coverage;
do not call a favorable transfer survivor proof of general cold-start acceleration.

## Bounded source-experience follow-up

**Proposed only; not implemented or launched.** After auditing the matched-client
study, one tractable extension is a predeclared source-experience imbalance. Keep
all clients on SPARC-like moderate dynamics and compare source budgets **40/40
versus 40/20 shots**, designating source B before seeing outcomes. Twenty-shot
learning is less experienced, not necessarily lower quality: a forty-shot learner
can deteriorate. This tests experience heterogeneity, not dynamics heterogeneity.

Retain local repair, reset-only, uniform, clipped and norm-matched uniform arms.
Use H=10 and stop at 50 total target shots, so the primary window is exactly the
first 40 shots after handover. Proposed calibration seed 127 and development
seeds 131/139 are separate from the other studies. Preserve common representation
seed 1729, paired physical cases, and identical untouched client prefixes across
conditions. Calibrate clipping once from an independent 40/40 source pool; freeze
the same thresholds for both conditions. If calibration is repeated instead of
reused, count both executions and verify their artifact hashes agree.

The primary interaction is the change in uniform-minus-local and
uniform-minus-reset cost AUC between source-budget conditions, paired by training
seed. Keep the same numerical tolerance and paired event-type comparison; report
absolute violations and all-failed flags. Clipping comparisons are secondary.
A small harmful contribution can evade norm clipping, so clipping is not a
contributor-quality detector. Four frozen-policy diagnostic episodes per source,
on a separate seed stream, describe source quality but never select contributors.
Sixteen final episodes per target arm assess the shot-50 endpoint, not persistence
to the earlier study's 120-shot horizon.

Without source/calibration reuse, the matrix costs **1,952 simulator episodes**:
two 80-shot calibrations, 280 development-source training shots, 40 shared target
prefix shots, 800 post-handover target training shots, 320 development evaluations,
320 final evaluations and 32 source diagnostics. Report transitions and optimizer
work separately because failures can shorten episodes. Source B still has the
same ten-shot warmup; only subsequent local experience changes.

Implementation would require explicit per-client source budgets, a distinct
calibration-budget contract, matching snapshot/revision metadata, and analyzer
coverage for both conditions. Existing frozen studies must remain reproducible.
Reject insufficient replay, invalid provenance or nonfinite state rather than
silently increasing budgets. Do not pick source B from observed returns, drop a
failed contributor or tune clip thresholds on target evaluation. If the matched
study shows pervasive control failure, diagnose the learner before this extension.

## Known actuator-capacity interpolation proposal

**Proposed only; not implemented or launched.** A smaller first variation than
changing transport physics is known auxiliary-heating capacity. Use one circular
SPARC-like plant with source maxima **25 and 20 MW** and a held-out target maximum
of **22.5 MW**; keep the ECRH maximum at 5 MW. Change only `aux_heat.hi` in immutable
per-client copies of the registered device. This tests interpolation of actuator
constraints, not changed plasma transport or detection of an unknown heater fault.

The current thermal adapter already includes physical actuator bounds and previous
powers in its 19-dimensional input; its critic converts normalized actions into
physical power using those bounds. The varied capacity is therefore observed.
This does not make the full plasma state observable: radial profiles and general
transport variation are not in those features. Preserve the same resolved absolute
SPARC moderate target ramp, tolerance, heating deposition, density, fusion,
disturbance configuration, solver and learner across capacities. Nominal-device
reachable bands cannot certify the altered action limits. A dynamic scripted/
feedback tracking preflight must establish feasibility under all three capacities,
including a smaller-time-step numerical spot check. If the 20-MW case cannot track
the common schedule, record that failure; do not silently relax its target.

Implementation must resolve environments and adapters by immutable client/regime
identity. The current cache uses registry device names, so merely labelling three
clients with different capacities would still give them the same nominal plant.
Freeze and hash the full resolved device, task, TORAX configuration, schema and
client mapping; reject changed limits on resume. Tests must verify the exact
20/22.5/25-MW endpoints and that only the intended field differs. The same physical
command should have the same plant response, while the same normalized command
has the declared different power. Existing fractional warmup therefore induces
different absolute-power coverage; log both units rather than concealing it.

Before a transfer claim, select one learner from local diagnostic evidence and
check local training across all capacities through the full 120-shot horizon.
Report the initial actor, fixed-power and predeclared feedback references. Early
improvement followed by collapse is not a stable baseline. Examine value probes,
critic gradients, replay coverage and possible omitted predictive state before
retuning aggregation. Better one-step prediction from profile/history features
would indicate missing predictive information, not by itself prove the cause of
an RL failure.

The subsequent transfer matrix can be limited to two fresh development seeds,
40 shots per source, H=10, 120 total target shots and three raw arms: local repair,
reset-only and uniform source transfer. Keep mix 0.25 and matched repair budgets.
Use the same primary 40-shot post-handover AUC and paired absolute limit-event
reporting; shot 120 is secondary. With four development episodes per arm at
10/20/30/50/120 and sixteen final episodes per arm, this is **840 training plus
216 evaluation episodes = 1,056**, excluding feasibility/local prerequisites.
The latter budgets, independent seed sets, selected learner and resolved physical
contracts must be frozen and charged before any launch. There is no clipping
calibration, contributor selection or fallback in this initial capacity study.
