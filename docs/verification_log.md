# Verification log

This log separates installation, deterministic checks, numerical experiments and
live research-agent behavior. A passing test suite or a model judge's acceptance
is not independent scientific verification. End-to-end research verification is
still in progress; the pending final run is not counted as a completed success.

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
| Full public app suite | 958 passed, 2 skipped in 26.30 seconds | Skips: absent private-checkpoint fixture and unavailable host symlink creation. Two existing Typer/Click warnings remain. Provider doubles in unit tests do not establish live research quality. |
| Windows publication and shadow-worker regression after the partial-run failure | 49 passed in 28.20 seconds | Injected transient/persistent publication failures, deadline limits and existing numerical worker tests; no learner settings changed. |
| Frontend logic suite | 33 passed | Exercises UI/HTTP contracts, public evidence selection and explicit shadow actions; no model inference or numerical experiment runs in these tests. |

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

The 22 recorded full attempts comprise 17 failed attempts, one model-accepted
report with independently identified defects, and four runs ending in explicit
insufficient evidence without a runtime error. Revisions must
preserve failed attempts and unsupported claims rather than hiding them.

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

- Complete and independently audit the bounded writer reflection, including raw
  versus gated citations, the baseline clipping setting, executed-tool scope and
  the complete 112-episode follow-up budget.
- Address the remaining centralized-versus-client privacy scope blind spot.
- Complete a full live question-to-search-to-extraction-to-reviewed-answer run,
  preserving its trace and any failed branches.
- Independently check the complete final claim set, including citation coverage,
  quoted context, exact numerical values and limits of applicability.
- Keep fresh-environment reproduction, developmental pilots and genuinely new
  paired-seed confirmation results separate in any presentation.

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
