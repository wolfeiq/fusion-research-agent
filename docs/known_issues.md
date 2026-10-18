# Known issues and verification limits

- Live model output can violate schemas or fail provider calls. Failures are retained; the agent must not relabel partial work as verified.
- A live report was accepted by the model judge but independent review found incomplete citation coverage and a centralized-vs-client privacy context error. Exact locator checks and stricter scope review have been added; live verification is ongoing.
- Fresh installation exposed a Click dependency conflict between legacy Flower 1.22 transport and TORAX/eqdsk. The CPU shadow runtime omits the unused transport extra and records its absence; original research environments are unchanged.
- Shadow experiments use in-process critic aggregation, not Flower network transport. They use shared laptop CPU, not a Nebius GPU instance.
- The two bundled one-seed pilots demonstrate executable mechanisms and preserve negative evidence. They do not establish reliable cold-start improvement.
