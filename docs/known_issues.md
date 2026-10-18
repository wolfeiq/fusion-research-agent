# Known issues and verification limits

- The latest full run (`cf248e70ab054c75b45e346c475d3e96`) returned insufficient evidence. Its shadow worker stopped after 7 episodes because Windows denied replacing the progress file. No factual claim survived review. Aggregation was not reached; clipping was unmeasured.
- Supported-claim preservation is implemented, but the latest repair exposed a report-envelope regression. Shared report decoding, malformed-limitations rejection and bounded Windows publication retries are now implemented and tested. A fresh live verification remains in progress.
- Proposal budgets must label evaluation subsets: 64 training + 48 evaluation = 112 episodes. Gate and initial evaluations are included in the 48. A model judge incorrectly double-counted those subsets. Proposed seeds were not established as unused across all histories.
- Exact quotes and locators do not establish semantic truth. A targeted citation-scope probe still exposes a centralized-versus-client privacy interpretation blind spot. Final scientific claims need independent source/context review.
- Model output can violate schemas or exhaust bounded budgets. Historical failures, unsupported claims, and the earlier model-accepted report with independent defects are retained in the verification log.
- Tested fixes cover bounded citation batches, limitations-field transport, consistent citation counts and unusable PDF excerpts. They do not establish universal model reliability.
- The fresh Windows CPU runtime omits unused legacy Flower transport because its Click requirement conflicts with TORAX/eqdsk. The transport test is skipped. Other platforms and CUDA have not been validated.
- Shadow trials use in-process numerical aggregation on shared laptop CPU, not Flower network transport, a Nebius GPU instance or a physical reactor.
- The recorded pilots and integration reproductions use development seed 17. They preserve negative evidence and exercise mechanisms; they do not establish reliable cold-start improvement.

See [the verification log](verification_log.md) for precise results and failure history.
