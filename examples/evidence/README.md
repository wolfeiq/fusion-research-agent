# Recorded starter evidence

These two terminal local CPU pilots are genuine recorded TORAX runs, not mock
data and not a new live run. Open the research view to inspect both comparisons.
The catalogue verifies the source archive, frozen configuration, result hashes,
terminal run record and deterministic comparison arithmetic before use.

Each six-file bundle deliberately contains summary-level evidence, not every
raw trajectory. Its verification does not perform a new trajectory audit.
The exact original source archive is retained for provenance; it is hashed, never
executed by the evidence reader. New shadow runs use the current repository code.
Do not edit a recorded result to match a preferred interpretation or newer code.

The learning-rate pilot tested a lower TD3 actor rate and recorded worse endpoint
metrics. The clipping pilot tests a separate bounded aggregation hypothesis;
read its comparison for its result. Neither one-seed pilot establishes efficacy
or cold-start speedup. Positive, null and negative outcomes remain visible.

Recorded identifiers, configurations and immutable byte hashes:

```json
[
  {
    "config": {
      "change": "td3_actor_lr",
      "rounds": 1,
      "schema": "fusion-shadow-config-v1",
      "seeds": [
        17
      ],
      "value": 5e-05
    },
    "id": "993df979e388405297f0195bfca504d6",
    "recorded_status": "completed",
    "result_sha256": "308ec8e3304189c229878995f7ecb6a5b632f63ea588817f69b6b9edbc528063",
    "source_zip_sha256": "9520548f718f756e75c8f30a83e76da4c5d7cc128eaca2d9cbc5097c05845a7c"
  },
  {
    "config": {
      "change": "clipping_initial_norm_fraction",
      "rounds": 1,
      "schema": "fusion-shadow-config-v1",
      "seeds": [
        17
      ],
      "value": 0.001
    },
    "id": "43d55b69d02440b3a019cf9b64f6845a",
    "recorded_status": "completed",
    "result_sha256": "e8f5403bf0bee4e983ca4294cfd6ed90f55bf57fbfe6329a64f9c82f6da8258e",
    "source_zip_sha256": "004a63037364f8b8adc54404bf364f669e86f0396bc553d71c292b5bd399fcbe"
  }
]
```
