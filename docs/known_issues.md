# Known issues and verification limits

- The latest full run (`4c57e0f656a443e98215f82f0aa11a4a`) finished with insufficient evidence, not a runtime error. Its retained primary-paper claim passed independent review; it did not produce a complete experimental answer.
- Experimental claims were correctly withheld for incomplete citation coverage. The correction pass damaged an already-valid citation, so its replacement was rejected. Preserving supported claims during bounded repair is being implemented and tested.
- Proposal budgets must label evaluation subsets: 64 training + 48 evaluation = 112 episodes. Gate and initial evaluations are included in the 48. A model judge incorrectly double-counted those subsets. Proposed seeds were not established as unused across all histories.
- Exact quotes and locators do not establish semantic truth. A targeted citation-scope probe still exposes a centralized-versus-client privacy interpretation blind spot. Final scientific claims need independent source/context review.
- Model output can violate schemas or exhaust bounded budgets. Historical failures, unsupported claims, and the earlier model-accepted report with independent defects are retained in the verification log.
- Tested fixes cover bounded citation batches, limitations-field transport, consistent citation counts and unusable PDF excerpts. They do not establish universal model reliability.
- The fresh Windows CPU runtime omits unused legacy Flower transport because its Click requirement conflicts with TORAX/eqdsk. The transport test is skipped. Other platforms and CUDA have not been validated.
- Shadow trials use in-process numerical aggregation on shared laptop CPU, not Flower network transport, a Nebius GPU instance or a physical reactor.
- The recorded pilots and integration reproductions use development seed 17. They preserve negative evidence and exercise mechanisms; they do not establish reliable cold-start improvement.

See [the verification log](verification_log.md) for precise results and failure history.
