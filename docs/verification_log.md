# Verification log

This log separates installation, deterministic checks, numerical experiments and
live research-agent behavior. A passing test suite or a model judge's acceptance
is not independent scientific verification. Live attempt 25 has separately
passed 35 independent reporting/integrity checks. This is a scoped workflow pass,
not an efficacy claim; all preceding attempts remain recorded. A separate
no-shadow negative check completed partially; its repeated check then failed
excerpt selection. After the excerpt-ID fix, check 28 separately passed 41
independent factual/reporting checks without new physics. These are two scoped
reporting passes, not scientific efficacy or general reliability guarantees.

## Fresh public-repository runtime

The public repository's root RL environment was installed afresh on Windows with
Python 3.12.13 and CPU execution. The resolved dependency set is recorded in the
hashed `requirements-rl-windows.lock`. The app retains its separate `flower-app/uv.lock`.
The Windows lock has not established Linux, macOS or NVIDIA GPU compatibility.

Installation exposed a real dependency conflict: legacy Flower 1.22 requires
`click<8.2`, while the current TORAX/eqdsk dependency requires `click>=8.3.2`.
The numerical shadow runtime does not use Flower transport, so its fresh install
omits the `fed` extra. The worker records `flwr: null` and
`flower_transport_used: false`. Required numerical dependencies remain mandatory.
The legacy Flower adapter is retained as reference code; it is outside this
validated CPU setup. The original experiment environment was not changed.

The reproduced numerical runtime recorded TORAX 1.4.3, JAX 0.11.1, NumPy 2.5.3
and PyTorch 2.14.1+cpu. A lock records dependency resolution, not a guarantee of
bit-identical outcomes on different platforms or hardware.

## Automated checks

| Check | Recorded outcome | Scope and remaining qualification |
|---|---|---|
| Initial public root suite | 251 tests passed; one failure exposed a missing calibration CLI file | The missing file was added. This initial run is not relabelled as an all-pass run. |
| Targeted follow-up after the export fix | 16 passed, no skips | This was a targeted rerun; it is separate from the subsequent full root-suite run. |
| Full public root suite in the freshly installed hashed Windows CPU environment | 252 passed, 1 skipped | The skip concerns unused legacy Flower network transport. The CPU shadow and numerical aggregation paths were tested. |
| Full public app suite before typed reporting | 958 passed, 2 skipped in 26.30 seconds | Skips: absent private-checkpoint fixture and unavailable host symlink creation. Two existing Typer/Click warnings remain. Provider doubles in unit tests do not establish live research quality. |
| Windows publication and shadow-worker regression after the partial-run failure | 49 passed in 28.20 seconds | Injected transient/persistent publication failures, deadline limits and existing numerical worker tests; no learner settings changed. |
| Frontend logic suite before typed reporting | 33 passed | Exercises UI/HTTP contracts, public evidence selection and explicit shadow actions; no model inference or numerical experiment runs in these tests. |
| Full public app suite with typed reporting and hypothesis quarantine | 1,043 passed, 2 skipped in 37.67 seconds; Ruff passed | Same two skips and existing Typer/Click warnings. Covers source-bound facts, exact excerpts, current/historical selection, failed workers, final recompilation and quarantine of optional unsupported references. Mock providers do not establish live quality. |
| Frontend logic suite with typed publication | 52 passed | Covers separate facts, attributed excerpts, untested hypotheses, rejected-hypothesis exclusion and proposal display as well as existing UI/HTTP contracts. |
| Targeted service-failure suite after `08dfbec` | 53 passed | Safe diagnostic redaction, bounded traceback locations, transient Windows replacement recovery, permanent failures and retained old state. |
| Full public app suite after `08dfbec` | 1,055 passed, 2 skipped in 26.09 seconds; Ruff passed | Same two skips and existing Typer/Click warnings. This supersedes the full-suite count above; separate invocations must not be added. |
| Configuration-card extension after `e720987` | 127 focused tests passed | Explicit baseline/candidate values for seven registered knobs, missing/invalid inputs and existing fact/publication boundaries. |
| Primary-source eligibility, writer and graph after `6496e1b` | 84 focused tests passed | URL/paper-path and extracted snapshot checks; secondary material cannot satisfy the required slot even with judge approval; missing eligible evidence remains insufficient. |
| Full public app suite after configuration and primary-reference fixes | 1,151 passed, 2 skipped in 27.60 seconds; Ruff passed | Same two skips and existing warnings. Frontend remains 52 passed. Prior invocations are retained but must not be added. |
| Immutable excerpt catalogue after `19d08c4` | 23 direct tests passed; independent focused run: 81 passed | Bounded catalogue, exact helper validation, immutable IDs, source hashes and original-question reconstruction. These invocations are separate from the full suite. |
| Full public app suite after `19d08c4` | 1,178 passed, 2 skipped in 27.53 seconds; Ruff passed | Two existing Typer/Click warnings remain. Frontend stays at 52 passed. This is automated coverage, not a new live pass. |

Counts describe separate invocations and must not be added as independent
coverage. Earlier test runs and failures remain documented in
[the research notes](../flower-app/RESEARCH.md). Missing transport coverage is
explicit; in-process aggregation tests do not exercise network federation.

## Fresh real TORAX reproduction

Trial `9b16eecbd8ea4077819201b91bdd58ac` completed in the newly installed public
repository environment with return code 0. It repeated the recorded clipping
hypothesis: `clipping_initial_norm_fraction=0.001`, one round, development seed 17,
with matched baseline/candidate configurations and local, reset-only, uniform and
clipped arms.

- 56 actual TORAX episodes completed: 32 training and 24 evaluation.
- The episode records account for 1,680 action windows.
- Recorded worker duration: 93.156 seconds on the local CPU.
- Source/configuration/result hashes and episode accounting were checked.
- Prior calibration is additional cost; it is not part of those 56 episodes.
- This reused seed 17. It is a reproduction, not an additional independent seed
  or a confirmation experiment.

The immutable worker-result SHA256 is
`f0f99a294061b7fa7b5adacbf67084099e8b6af1b0942720439d84eface80f95`.
The frozen-source archive SHA256 is
`2a222f6d9707f2db44cf443432158652a9ee54bda6ffb7a3f0e82e0be1ab6d7c`.

This run verifies that the fresh CPU environment can execute the bounded
numerical experiment without legacy Flower installed. It does not establish
reliable federation benefit, shots-to-competence improvement, NVIDIA acceleration
or network-transport correctness. The worker's follow-up recommendation remains
a proposal. No existing controller is automatically changed or deployed.

The [published starter bundles](../examples/evidence/README.md) retain the earlier
negative learning-rate pilot and clipping pilot. Private runtime state, provider
credentials and private logs are not part of those public bundles.

## Live research-agent audit

Live Token Factory/Tavily checks and the adversarial judge exercise are separate
from the full multi-agent research workflow. They can demonstrate real provider
calls, extraction or a particular rejection without proving that every final
report is supported.

The documented public-bundle adversarial judge check passed live in 5.468
seconds. It retained the correct single-seed cost-increase observation and
rejected the inverted interpretation and universal reactor-benefit claim.
Deterministic verification separately rejected the invented quotation. This
bounded check does not resolve the distinct privacy-scope blind spot below.

| Research attempt or probe | Recorded state | Interpretation |
|---|---|---|
| Full attempts 1–12 | Failed; retained in the audit history | Failures included provider/transport, schema, context and citation problems. A completed shadow job inside a failed research attempt remains numerical evidence, not a successful complete research answer. |
| Full attempt 13 | Accepted by the model judge; independent review found defects | Citation coverage was incomplete and some supporting context was interpreted with the wrong scope. This is not an independently verified report. |
| Full attempt 14 | Failed; retained in the audit history | It is not excluded from the attempt history or counted as successful. |
| Full attempt 15: `5dd98f734aee491eb92a8c0676dd7c5d` | Failed: `Agent response exceeds the tool batch limit` | The writer emitted more than eight tool calls in a response. Bounded writer batch handling was subsequently fixed in commit `7f6b8c2`; this historical failure is retained. |
| Full attempt 16: `f3bf4111c7bd4598bb23b1a804135ba2` | Finished with `insufficient_evidence`; no runtime error | Recorded 26 model calls, 521,796 tokens, 2 searches, 2 page extractions and 34 tool calls. It ran a real shadow pilot, but independent review still found citation/scope and proposal-accounting defects. |
| Targeted live citation/scope probe after an architecture fix | C1 and C2 were rejected; the C4 scope defect remained | These are validation-fixture identifiers. The remaining blind spot concerns interpreting centralized differential-privacy evidence as support for client-level protection in a federated setting. The probe is not an all-pass judge validation. |
| Full attempt 17: `5e46c66a6d1d4395a65b23c189e14a0c` | Failed before the judge: `next_run.limitations must be a string` | 26 model calls and 464,837 tokens. The real shadow completed, but no valid report reached independent model judgment or the new writer reflection. |
| Full attempt 18: `a65ff29c4c554f999afeee244a8e9800` | Failed before the judge: no traceable experiment claim | 26 model calls and 547,348 tokens. Three oversized citation-helper batches (14, 9, 12 calls) were rejected under the eight-call limit; the writer then invented dotted locators. Deterministic citation checks rejected those claims. |
| Full attempt 19: `7a5ab3531a2046579e615c8c4a1b75fd` | Failed before the judge: PDF quote contained control characters | Citation helpers now executed, but one returned a canonical PDF excerpt containing U+001A/U+001B that the report validator rejects. Earlier drafts also exceeded the former eight-citation-per-claim limit. |
| Full attempt 20: `4c57e0f656a443e98215f82f0aa11a4a` | Finished with `insufficient_evidence`; no runtime error | 32 model calls, 763,616 recorded tokens, one search, three extractions and 36 tool calls. Independent audit approved the retained primary-paper claim, but experimental claims lacked citation coverage. One reflection damaged a valid web reference and was rejected. |
| Full attempt 21: `cf248e70ab054c75b45e346c475d3e96` | Finished with `insufficient_evidence`; shadow worker failed | The worker encountered Windows error 5 replacing `progress.json` after 7 completed episodes. The candidate and aggregation were not reached. No factual claim survived review; some rejected/ancillary prose incorrectly described unmeasured clipping as inactive. Reflection also exposed inconsistent report-envelope handling. |
| Full attempt 22: `843402cd5dd44c9d9126ea5314ffe2b3` | Finished with `insufficient_evidence`; shadow completed | 26 model calls, 563,828 recorded tokens, one search, two extractions and 30 tool calls. The real worker completed all 56 episodes. Reflection executed and complete-record citations helped, but independent review rejected two claims retained by the model judge. |
| Full attempt 23: `90c553019d654f2c966882cd10d240fd` | Failed: writer could not select a valid evidence packet within its bounded budget | The new typed publication path did not produce a report because optional hypotheses referenced unread or unknown sources. The shadow completed 56 episodes in 98.468 seconds; all 57 protected source hashes remained unchanged. |
| Full attempt 24: `9bcfc668aecd4b17a2e8608882747e1f` | Failed while processing a response | No report or current shadow source was recorded. No traceback was retained; the root cause remains unproven. Later safe diagnostics and replacement retries do not retrospectively diagnose this attempt. |
| Full attempt 25: `2208045d8aea469690a67d442dec46c2` | Completed; passed 35 independent reporting/integrity checks | All seven roles used Nebius inference with Tavily search/extraction and one real shadow. Recorded 22 model calls, 419,110 tokens, two searches, three extractions and 17 tool calls. The source-bound report passed audit; this is not scientific efficacy evidence. |
| No-shadow negative check 26: `e1f7f92987e14e8bbb80b24aa1125da7` | Completed partially; not a full independent pass | Four unfavorable cost differences and keep-baseline advice were correct. The exact quotation came from TinyCTO secondary coverage; baseline learning rate and uniform/clipped competence details were missing. |
| Repeated no-shadow negative check 27: `58b5629613d94afebdbcbb5583c08057` | Failed; no report | Primary papers were extracted, but all three writer selections failed with `The entire anchor must fit within max_words`. No new physics ran. |
| No-shadow negative check 28: `6792cb0daad24ac38f12c717755d320b` | Completed; passed 41 independent factual/reporting checks | 12 exact saved-evidence facts, complete learning-rate/cost/competence coverage, one exact primary TD3 excerpt and keep-baseline advice. Its improvement hypothesis was critic-rejected. No new physics ran. |

Across all 28 completed research runs, including the no-shadow negative checks,
there are 20 failed runs, two completed reports with independent defects
(the earlier model-accepted attempt 13 and partial check 26), four explicit
insufficient-evidence outcomes without a runtime error, and two independently
audited reporting passes (25 and 28). The passes are scoped to their actual
workflows: 25 includes a new shadow, while 28 reads a saved unfavorable pilot.
Neither pass erases earlier failures or establishes autonomous scientific reliability.

### Attempt 16: completed execution, incomplete scientific answer

Its shadow trial `133a68b3434342b99586ae52a26c41e0` completed 56 actual TORAX
episodes in 93.5 seconds: 32 training, 24 evaluation and 1,680 action windows.
It repeated development seed 17 and one round with clipping fraction `0.001`.
All 57 captured source/calibration hashes remained unchanged. This establishes
another execution of the isolated pilot, not an additional independent training
seed, confirmation result or improvement to the existing experiments.

Independent review of the retained report found:

- The final claim C1 cited a deployed/gate-selected field while asserting a raw
  learner outcome. Raw and gated branches require their own exact references,
  even where their recorded numerical values happen to agree.
- The baseline clipping setting `0.1` appeared without its own supporting
  citation. A quoted result value does not support every contextual assertion
  in the surrounding sentence.
- A stale limitation said the agent could not launch a trial even though this
  run actually launched and completed its authorized isolated shadow pilot.
- The proposed two-seed baseline/candidate follow-up needs **112 total episodes:
  64 training and 48 evaluation**. Counts must include both configurations and
  their source, target and evaluation work; raw and gate-selected metrics remain
  separate. Prior calibration is additional cost.
- Literature support must be checked against extracted primary papers and their
  actual experimental/privacy setting, not inferred from general summaries.

A bounded writer-only reflection was implemented and tested in commit `31222b7`.
It repairs from captured evidence and reserves a fresh independent judge, without
new retrieval or another physics trial. Attempt 17 failed earlier, before it
could reach this path; offline test coverage is not a live reflection result.

### Attempt 17: successful physics, rejected report format

Shadow `d7d4005eb6544eb186f3147b1eb49bb7` completed 56 episodes in 91.141
seconds. Independent audit checked all 57 frozen/current source hashes, the ZIP,
raw result hashes, comparison and captured telemetry. The cost values reproduce
the same development seed; all competence flags remain false. The trace contains
one shadow attempt and one generic public literature query.

The writer exhausted its bounded turns after a proposal prose field failed type
validation. No rejected model payload is retained in the trace, so the exact
malformed value cannot be independently reconstructed. A subsequent formatting
adapter is limited to documented prose fields; numerical, citation and semantic
validation remain mandatory.

Exact quotations and JSON pointers check traceability. They do not by themselves
prove entailment, applicability, numerical interpretation or scientific scope.
A model can quote a real passage while applying it to a different setting. Final
claims need both source-level checking and an independent review of what the
source actually establishes. A proposal must also distinguish an observed
experiment outcome from a literature-backed hypothesis for a future run.

### Attempt 18: citation batching blocked the writer

Shadow `571f8d4aba7e431796520cf287da0e90` completed 56 episodes in 92.406
seconds. The independent audit checked the 57 source files, frozen archive,
raw result, comparison and captured telemetry. Its 32 training and 24 evaluation
episodes again reproduced development seed 17, with no competence successes.

The formatting adapter passed its regression tests, but this run exposed another
transport limit: repeated citation lookup batches exceeded eight calls and were
rejected in full. No citation helper from those batches executed. The writer
then supplied dotted paths and partial-object quotations; the deterministic
verifier correctly refused them. Commit `3e46760` allows up to 24 calls in writer-only batches containing solely
read-only citation helpers. Other batches retain eight. The global tool-call
budget and exact citation checks remain, with terminal-report capacity reserved. This failure is not an accepted or independently verified report.

### Attempt 19: PDF citation helper and report validation disagreed

The real shadow completed 56 episodes in 91.938 seconds, with all 57 source
hashes and captured numerical evidence checked. No competence success was
recorded. The citation helper returned a 20-word excerpt from the primary PDF
whose captured text contains U+001A/U+001B. Those bytes were authentic to the
snapshot, but the report validator forbids them. The fix rejects unusable
excerpts before returning a citation, leaving canonical source bytes unchanged.

The model also needed nine citations for a detailed claim. Commit `ae4e5a0`
raises that bound to 16 consistently in the advertised schema and validator;
17 remains rejected and every citation still needs exact support. Existing
report-size, global call, and web-quotation limits remain unchanged. These fixes
do not retroactively turn attempt 19 into a successful report.

### Attempt 20: correct withholding, incomplete experiment synthesis

The isolated shadow completed 56 episodes in 94.078 seconds. Independent audit
again verified all 57 protected source files and the raw, comparison and captured
evidence hashes. Its repeated development seed does not add independent efficacy
evidence.

The retained primary-paper claim passed independent contextual review. The judge
correctly excluded the experimental claims for uncited settings, cross-role
assertions and endpoint/control values. The single reflection attempted to
repair them but also rewrote an already-valid web citation incorrectly; exact
validation rejected its replacement and retained the earlier insufficient report.

The follow-up proposal still needs explicit accounting: 64 training plus 48
evaluation equals 112; 16 gate and eight initial evaluations are subsets of the
48, not extra costs. The model judge incorrectly added these subsets again.
Claims that proposed seeds are unused were not verified against all histories.
Trajectory absence must be scoped to the selected snapshot: the worker saved
per-episode traces. The stop reason also overstated citation coverage. None of
these proposal/ancillary statements should be treated as independently approved.

## Work still required

- Preserve the two scoped live audits and all earlier failures as regression
  evidence. Check 26 remains partial and check 27 failed despite check 28's pass.
- Broaden robustness beyond the one independently audited live workflow: varied
  questions, irrelevant literature, unavailable sources and failed workers must
  remain explicit rather than becoming invented findings.
- Preserve the publication boundary as features change: current/historical
  provenance, exact numerical values, clipping identities, raw versus gated
  outcomes, quoted context and proposal accounting remain review requirements.
- Use the new safe diagnostics if a response-processing failure recurs. The old
  traceback-free attempt 24 cannot be retroactively assigned a root cause.
- Keep developmental reproductions and genuinely new paired-seed confirmation
  results separate. Exact source quotation does not establish applicability,
  a causal mechanism or a scientific federation benefit.

See [reporting integrity](reporting_integrity.md) for the default boundary.
No statement in this log establishes production reactor readiness or a reliable
cold-start advantage from federation.

## Attempt 21: a retained worker failure

Shadow `ed8410c85b8d4be8a102729100d271c3` completed 7 episodes (5 training, 2 evaluation; 210 action windows) before Windows denied replacing its progress file. An eighth episode has a start marker but is not counted as completed. Duration was 28.391 seconds. All 57 frozen/current source hashes and saved result/comparison hashes matched. The trace cannot identify which file handle caused the denial.

Aggregation and candidate evaluation were not reached. Clipping activation and a paired cost difference were **not measured**; this is neither an RL failure nor evidence that clipping was inactive. The model output retained this distinction incorrectly in rejected claims and some ancillary text. The actual answer withheld all four claims. The output is not an independently verified research report.

A separate reflection regression rejected a non-native report envelope even though the ordinary report path supports bounded legacy envelopes. The exact submitted envelope was not retained, so its particular shape cannot be reconstructed from the error. The required fix is shared decoding and strict validation, not inventing missing fields.

The publication repair is commit `4688225`: atomic replacement retries only Windows access/sharing/lock errors 5/32/33, for at most six attempts and 0.31 seconds scheduled delay, bounded by the original worker deadline. Permanent errors still fail and retain partial outputs. Commit `89f0228` shares existing report-envelope decoding between initial writing and reflection, rejects malformed container-shaped limitations, and recommends small complete citation objects rather than disconnected scalar values.

## Attempt 22: worker completed, semantic review still failed

Shadow `5818a87a8703477c9e29d1ba3dca4e40` completed all 56 episodes in 92.437 seconds after the Windows publication repair. It repeats development seed 17; this is integration reproduction, not another independent efficacy observation. The research workflow finished without a runtime error and executed its bounded reflection. No complete experimental research report is counted as independently verified.

The retained answer still has two material citation defects. Its endpoint-cost claim gives absolute candidate/baseline costs without citing those values; the model judge improperly excused this as arithmetic behind cited differences. Its literature claim uses a centralized sample-level gradient-privacy paragraph to support federated client-update clipping and client-level privacy. The judge explicitly noticed the centralized context, then wrongly rescued the claim using the paper's overall subject. Exact quote matching and a second model context therefore do not establish semantic support. The complete-object citation guidance reduced mechanical gaps but did not solve this judgment failure.

Independent integrity checks matched all 57 frozen/current source files, source archive, raw/canonical result hashes and comparison/diagnostic projections. Only `scripts/run_shadow_trial.py` changed versus attempt 21; configuration, baseline and calibration were unchanged. Accounting was 32 training + 24 evaluation episodes and 1,680 windows.

Ancillary limitations remain unreliable: the model says the gate accepted a fallback, but it accepted the candidate; the stop reason calls the excluded clipping claim supported. The final answer cites historical `E001`, not the new shadow snapshot, so it must not be represented as a freshly sourced result. The proposed 56-episode budget is correct. These qualifications do not change the independent report verdict: fail.


## Attempts 23 and 24: typed publication and retained runtime failures

The default reporting architecture now selects server-authored fact IDs and exact
source excerpts. The final publication is rebuilt from the captured evidence;
free model interpretations and a model judge's approval cannot author measured
claims. Current-run and historical provenance are explicit. Proposal budgets are
validated against the fixed shadow protocol. This change responds to the scope
errors preserved in attempt 22; it does not erase them.

Attempt 23's real shadow again completed 56 episodes in 98.468 seconds, with all
57 protected source hashes checked. The writer used invalid source references
in optional hypotheses, and its bounded selection attempts failed. No report was
published. Commit `6027357` now quarantines those optional interpretations,
records `hypothesis_rejected` events and excludes them from the displayed
hypotheses. References are not guessed or repaired. Factual selections, citations
and proposal checks remain strict. Repeating seed 17 is integration reproduction,
not independent evidence of federation efficacy.

Attempt 24 failed while processing a response before a current shadow result or
report was recorded. Its generic error has no retained traceback; the root cause
remains unproven. A transient Windows file lock is a hypothesis, not a finding.
The later attempt 25 passed a separate independent publication audit, recorded
below. Local automated checks alone do not substitute for that audit.

Commit `08dfbec` adds safe observability for future failures: exception class,
up to 12 bounded filename/function/line records and numeric OS error codes.
Exception text, full paths, source text, provider payloads and locals are omitted.
The service retries Windows access/sharing/lock replacement errors 5/32/33 for
at most six attempts and 0.31 seconds of scheduled delay per replacement. It
never deletes the previous state to force a write. Permanent or exhausted errors
still fail. The 53 targeted service tests and the later 1,055-test app suite
verify those code paths; they do not prove that a file lock caused attempt 24.


## Attempt 25: independently audited live reporting pass

Run `2208045d8aea469690a67d442dec46c2` completed the full question-to-retrieval-to-
shadow-to-publication workflow. All seven LangGraph roles used Nebius Token
Factory, and Tavily performed two searches and three page extractions. The
recorded usage was 22 model calls, 419,110 tokens and 17 tool calls. The final
report passed 35 independent reporting and integrity checks.

Shadow `b4838455d2a0460d8007314be78af07a` completed all 56 real TORAX episodes in
93.469 seconds: 32 training plus 24 evaluation, with 1,680 action windows. All
57 protected source files matched. The audit recomputed all endpoint mean costs
exactly from saved rewards and checked the source-bound numerical evidence.
The run repeats development seed 17; it is integration reproduction, not an
additional independent cold-start or federation efficacy observation.

The publication contains eight server-authored fact cards. Seven are bound to
the current shadow; the remaining gate observation is explicitly labelled as
historical. Its single literature excerpt is an exact attributed 25-word
quotation from the primary PMLR paper, without an invented paraphrase claiming
broader applicability. The report submitted no hypotheses, and no model prose
was promoted into a factual claim.

The next-run proposal comes from the current source and remains a proposal:
112 episodes = 64 training + 48 evaluation. Gate (16), initial/fixed (8) and
report (24) episodes are disjoint subsets of evaluation, not extra episodes.
Prior calibration and optimizer work remain separate. No policy was automatically
activated and no recommended follow-up was launched by publication.

This is a scoped pass for this live reporting workflow. It does not establish
universal agent reliability, simulator validity, reactor safety, applicability
of every cited method or a reliable cold-start benefit. The prior 24 attempts
remain unchanged in the failure history. A public, explicit projection is
available in [reporting_check.json](../examples/verification/reporting_check.json):
two local source snapshots, eight facts, 35 checks and one 25-word attributed
excerpt. It is not the full decision trace and does not reproduce bulk web text.


## Negative check 26 and configuration/primary-reference fixes

Run `e1f7f92987e14e8bbb80b24aa1125da7` read the saved unfavorable learning-rate
pilot without launching a new shadow. It correctly reported cost increases in
all four paired arms and recommended keeping the baseline. Its exact literature
excerpt came from TinyCTO secondary coverage, and the selected facts omitted
the baseline learning rate and the uniform/clipped competence flags. The report
was therefore only a partial answer, not a complete independently accepted pass.

Commit `e720987` adds typed configuration cards for all seven registered knobs,
including explicit baseline/candidate change values, and permits up to 12
selected fact cards. It passed 127 focused tests. No missing values are guessed.

Commit `6496e1b` checks recognized primary-publication hosts and paper paths,
plus actual extracted, hash-bound snapshots. Secondary coverage cannot satisfy
the required literature slot even when a model judge accepts it; an absent
eligible primary excerpt keeps the report insufficient. An already discovered
eligible page may be extracted on the literature agent's final turn without
adding model turns. The 84 focused eligibility/writer/graph tests passed.
The check is a conservative URL/location heuristic, not certification of
authorship, peer review, quality or scientific applicability.

The full suite after these changes passed 1,151 tests with two skips and two
existing warnings in 27.60 seconds; Ruff passed and the frontend remains at
52 passed. The same-question negative check 27,
`58b5629613d94afebdbcbb5583c08057`, subsequently failed without producing a report.
All three bounded writer selections returned `The entire anchor must fit within
max_words`. Retrieval had succeeded: extracted primary sources included the TD3
PMLR HTML/PDF and Henderson arXiv paper. The failure occurred in constructing a
valid excerpt selection, not in obtaining primary literature. No new physics ran.

Commit `19d08c4` subsequently implemented a server-built catalogue of exact,
immutable excerpt IDs. The writer selects at most one offered entry instead of
constructing quote text, anchors or offsets. Each entry is hash-bound and passes
the exact citation helper. The original question is preserved for deterministic
reconstruction. Catalogue generation is bounded to 12 entries per source and
48 globally, with at most 128 candidates and 24 helper calls per source.

The catalogue passed 23 direct tests and an independent 81-test focused run.
The full app suite passed 1,178 tests with two skips and two existing Typer/Click
warnings in 27.53 seconds; Ruff passed, and frontend coverage remains 52 passed.
These separate invocations must not be added together as independent coverage.
Same-question check 28 subsequently passed an independent audit, described below.
Check 27 remains failed in the aggregate counts.


## Check 28: independently audited unfavorable-result reporting pass

Run `6792cb0daad24ac38f12c717755d320b` completed the same no-shadow question that
checks 26 and 27 did not fully answer. It passed 41 independent factual/reporting
checks. Recorded usage was 18 model calls, 248,374 tokens, three Tavily searches,
four extractions and 15 tool calls. There were zero evidence-selection rejections.

All 12 published fact cards matched the explicitly labelled saved pilot
`993df979e388405297f0195bfca504d6`. They include the actual TD3 actor learning-rate
change from 0.0001 to 0.00005, all four positive candidate-minus-baseline raw cost
differences (the candidate was worse), and false/false baseline/candidate
competence flags for local, reset-only, uniform and clipped arms. No favorable
claim was substituted for this negative result.

The literature card is an exact 25-word excerpt from the actual Fujimoto TD3
PMLR PDF, relevant to fair multi-seed evaluation. It does not establish actor
learning-rate efficacy. The judge marked the one model-generated improvement
hypothesis `critic_rejected`; it remained unverified and never entered the
factual answer. The exact source excerpt is distinct from that rejected hypothesis.

The recommendation keeps the baseline. Its unlaunched 112-episode follow-up
proposal would repeat the same losing candidate on development seeds 29 and 43
for diagnosis: 64 training + 48 evaluation, with 16 gate, 8 initial/fixed and
24 report episodes inside evaluation. It neither approves adoption of the
candidate nor establishes those seeds as globally unused. No new shadow,
physics, policy activation or change to current experiments occurred.

The [public negative verification record](../examples/verification/negative_reporting_check.json)
preserves this scoped audit alongside the positive clipping reproduction. It
contains 12 facts, 41 checks, manual review, the proposal and one 25-word paper
excerpt, with no added experiment source snapshots. Each fact/proposal quotation
matches the already-public pilot comparison at commit `f0a0174`; the artifact
records its fixture path, commit and SHA. It introduces no new experiment
metrics or private paths. The literature quotation ends mid-sentence at “perform”
and is explicitly an excerpt. The rejected hypothesis and unverified gap notes
are not factual findings.

This pass verifies the reporting of saved unfavorable evidence. It is not an
independent RL replicate, a federation efficacy claim or a guarantee of correct
autonomous scientific decisions for every future question.

## Three additional live workflow trials: frozen protocol

The next three cases test different bounded controller changes. Their questions,
configuration values, evidence selections and request hashes were frozen before
launch. Outcome signs are not selected in advance. Each uses the same SPARC-like
simulation, development seed 29 and one federation round, with fresh controllers
and paired local, reset-only, uniform and clipped arms for both configurations.
These are three configuration datasets, not three reactors or independent seeds.
Seed 29 was previously used in development and is not confirmation data.

| Case | Changed setting | Baseline | Candidate |
| --- | --- | --- | --- |
| SAC actor learning | `sac_actor_lr` | 0.0003 | 0.0001 |
| SAC critic mixing | `sac_mix` | 0.25 | 0.1 |
| TD3 action coordinates | `heating_critic_coordinates` | `normalized` | `physical` |

The frozen budget is 56 episodes per case: 32 training and 24 evaluation, up to
1,680 action windows and a 600-second worker limit. For all three cases this is
168 episodes, 96 training, 72 evaluation and up to 5,040 action windows. These
are planned budgets; actual work and incomplete attempts must be audited
separately. Earlier calibration and replay-only optimizer work are additional
costs. A lower candidate endpoint cost does not by itself establish a federation
advantage over local learning or faster shots-to-competence.

Each question requests the current status, exact changed setting, actual
accounting, all four paired raw endpoint differences, all four competence
comparisons and available candidate SAC clipping evidence. The report must
distinguish its new shadow from saved evidence, preserve negative and inconclusive
outcomes, and keep raw learner and gate fallback separate. It must retrieve and
read a relevant primary publication through Tavily. One exact excerpt supplies
methodological context; it does not prove the numerical change helps. Optional
explanations remain untested. A reported next-run proposal is not launched or
activated automatically.

The first SAC actor request, `3ed89057e8d34f16a8eb63ff003cb59f`, was intentionally
interrupted before any `shadow_start` event or worker lease. Its question
requested seed 29, while the old agent tool fixed seed 17. This was an interface
contract failure caught before physics, not a numerical trial. The interrupted
attempt is retained separately from the preceding 28 terminal reporting checks.

The corrected agent tool requires an explicit integer seed from 0 through
1,000,000. Missing, boolean, fractional, string, null or out-of-range seeds fail
before consuming the one-attempt allowance or starting a worker. The event
records the validated seed and one-round limit; the same values enter the frozen
manifest, worker configuration and returned evidence. Only direct historical
service callers retain the default seed 17. The change does not expose extra
rounds or multiple seeds to the agent and leaves the 56-episode ceiling unchanged.
The engine/service/graph/reflection focused suite passed 96 tests; Ruff passed.
Those tests use synthetic subprocesses and do not constitute live physics results.

The three corrected live cases remain pending until their execution and reports
are independently audited. No outcome, deployment or efficacy pass is implied by
this frozen protocol.


## Three additional live workflow trials: completed physics, reporting audit open

The three frozen seed-29, one-round SPARC-like datasets completed: SAC actor
learning rate 0.0003 to 0.0001, SAC critic mixing 0.25 to 0.1, and TD3 critic
coordinates normalized to physical. Actual work was 56 episodes per dataset,
168 total: 96 training, 72 evaluation and 5,040 action windows. Seed 29 is
previously used development data; the three configurations are not independent
training-seed confirmations. Earlier calibration and replay-only optimizer work
are separate costs.

The [three-trial audit](three_trial_audit.md) links all saved comparisons and
separates candidate-minus-baseline raw arm costs from candidate federation versus
local. Candidate uniform-minus-local raw endpoint costs were +0.00367020,
+0.01578276 and -0.00476015 respectively (rounded cost differences, not percentages).
All baseline/candidate raw competence flags were false across all four arms in
all three datasets. Every recorded recommendation was keep_baseline. These
results do not demonstrate a general federation or shots-to-competence advantage.

The initial SAC actor report passed 62/63 independent checks but omitted local
competence and used a generic literature quotation. The SAC critic-mixing report
was insufficient at 59/63: local cost and reset-only competence were missing,
and the retrieved mirror did not provide an eligible primary reference. The TD3
physics evidence passed 38/38 checks, while its report failed three writer
attempts involving malformed optional gaps, missing paired endpoint coverage and
selection beyond the 12-card limit. None is recorded as a full initial workflow
pass.

Repairs introduced a mandatory current-source comparison bundle, source-bound
recommendations, explicit quarantine of malformed optional annotations and
bounded primary-source recovery. Three live report-only repairs now include the
12 required measured fact cards and reused the completed datasets, adding zero
physics work. The SAC critic-mixing repair still accepted an exact but irrelevant
metadata excerpt. That defect is retained; excerpt-catalogue relevance is being
fixed and independently verified. Three full end-to-end passes are not yet
established. This newer audit updates the pending status in the frozen-protocol
entry above, without deleting its pre-launch record.

The [public dataset index](../examples/evidence/3demo-datasets.json) records the
six exported files per dataset and their hashes. Five files retain exact trial
bytes; run.json is a process-metadata-free projection. EvidenceCatalog verified
the hash chain and recomputed comparison arithmetic; all 57 source members per
archive match the recorded manifest and already-public source at commit
2837fac587a87a07a731d8de199ef32db6df7eb6. That source match is not a claimed launch
commit or independent physical replication. No provider transcripts, bulk web
text or runtime logs accompany these numerical bundles. Original experiments,
failed reports and numerical evidence remain unchanged; no policy or follow-up
experiment has been automatically activated.

## Three reporting-only repairs: scientific excerpts and scoped factual checks

The completed seed-29 shadow datasets were reused for bounded reporting-only
repairs. Attempt 1 failed on provider connection before successful inference;
attempt 2 returned all measured cards but selected bibliographic metadata for
the SAC critic-mixing case. Both attempts remain preserved. The latest attempt
does not retrospectively convert the original partial, insufficient or failed
research runs into full workflow passes.

The excerpt catalogue now excludes identifiable frontmatter, navigation and
bibliography. When a scientific sentence is repeated only in excluded material
such as BibTeX, a server-selected original locator may disambiguate it. The
locator must match an exact citation-helper candidate, be wholly within the
retained scientific body and identify an anchor unique in that body. Publication
revalidates it. Repeated scientific-body text, forged offsets and control
characters remain invalid; source hashes and the 25-word bound are unchanged.
The final public app suite recorded **1,277 passed, two skips and two existing
warnings**. Counts from prior invocations are not added to this total.

Attempt 3 completed reporting-only repairs for research runs
`60c75c21a2c74c899d231159a72ec9a3`, `aaa68592d5da4e37a9cb57eaf4d6f0dc` and
`8332c5c2279e417fa9f86543a6b1c941`. Independent mechanical audits passed **37/37
checks for each**. The runs used **3, 2 and 2 model calls** respectively, with
zero new searches, zero page extractions and zero additional physics episodes.
Each answer contains twelve exact server-authored current-trial facts, all
four paired raw cost differences, all four competence pairs, measured clipping,
and the exact keep-baseline recommendation with an unlaunched 112-episode
proposal. Original research records, numerical files and local source snapshots
were confirmed unchanged. No shadow executor was available to these repairs.

Manual review found scientific prose in all three excerpts, with explicit scope
limits. The SAC reference describes the reward/entropy objective; the TD3
reference discusses overestimation bias. Neither explains the measured change
causally. The SAC critic-mixing answer selected HAPFL, whose surrounding text
uses PPO to manage client model allocation and training intensity. That is
adjacent context, not direct evidence about federating SAC critic parameters.
The metadata-selection defect is resolved, while scientific applicability still
requires review.

Optional model gap notes still incorrectly call the recorded inactive SAC q0
clipping evidence missing in the actor-learning-rate and TD3 cases. The TD3
judge corrects this; the actor-learning-rate judge repeats the confusion. The
authoritative fact card correctly records zero of three clipped contributors
and all factors 1.0. All three model causal hypotheses were `critic_rejected`;
none entered the factual answer. The [updated three-trial audit](three_trial_audit.md)
records these remaining interpretation limits. This is three scoped factual
publication checks using preserved data, not three complete workflow successes,
new independent RL replications or evidence that federation solves cold start.


## Research-only interface and saved-data recovery

The app now uses a cream/red editorial theme and exposes only research, saved
TORAX datasets, findings and explicitly opt-in experimental beta generation.
Legacy Flower/SuperGrid surfaces are hidden/inert and their scripts are not
loaded. The focused TD3 critic-coordinate comparison is selected by default;
the other two SAC datasets remain available. Dataset selection is not a promise
of successful inference or a federation benefit.

A deployed live saved-data audit (`18d40967f5dc461a8aa0a4ded53b8f1d`) completed
with exact numerical cards and a relevant TD3 primary-paper excerpt, but omitted
the clipped competence card and recorded recommendation. That partial audit is
retained. The complete 11/12-card requirement now also applies to one validated
saved single-seed comparison, preserving its historical provenance and recorded
proposal. Multiple datasets/seeds are not silently pooled. Recompiling the exact
failed selection restored the missing evidence without another simulation.

Browser testing also exposed an offline-format mismatch: local inspection
returned the legacy report format, so the new UI could not render its measurements.
One-at-a-time inspection of all three published datasets now uses the same typed
fact boundary, without inference, retrieval, judging or training. Browser QA
confirmed 12 cards and an exact-value citation dialog for the TD3 dataset.
Desktop and 390-pixel responsive views were inspected; the temporary viewport
was reset afterward.

Graceful recovery now includes a 20-second fetch timeout, status reconciliation
after an uncertain POST, explicit retry with physics disabled, retained reports
and trace with original run IDs, and safe error categories. Backend credentials
are redacted before error truncation. Failed terminal persistence preserves the
current export and blocks replacing unsaved evidence until storage recovers.
Original failures and failed/partial runs are not relabelled as successes.

The combined local suite passed **1,308 tests**, with **2 skipped** and **2 existing
Typer/Click warnings** in 40.50 seconds. All **65 frontend tests** and focused Ruff
checks passed. The credential scan found no actual configured provider-key values
in tracked files, Git history or archived source members. These checks are separate
from the live Nebius/Tavily audit and do not validate Linux TORAX execution.

## Final deployed saved-data audit

The second cloud verification, run `92753c1d8f4b49d5b3e62296d98536dc`, completed
on deployed revision `94ad4ba` with **44/44 verification checks passing**. It
used the frozen TD3-only question, live Nebius inference and Tavily retrieval,
with shadow generation disabled. All 12 saved fact cards, exact source snapshots,
citation endpoints, the recorded recommendation and full proposed budget passed.
No new physics episodes ran; existing app/worker experiment state was unchanged.
The first cloud audit and its omissions remain preserved and are not overwritten.

Independent excerpt review identifies the scientific abstract of Fujimoto et al.,
*Addressing Function Approximation Error in Actor-Critic Methods*, PMLR 80 (2018).
Its value-estimation-error context is relevant to critic methodology, not direct
causal evidence for coordinate normalization or federated cold-start gains.
The exact tested settings/question are in [verified_demo.md](verified_demo.md).
The code revision's Linux research-image and real-Caddy CI checks also passed
([run 39141449138](https://github.com/wolfeiq/fusion-research-agent/actions/runs/39141449138)).
