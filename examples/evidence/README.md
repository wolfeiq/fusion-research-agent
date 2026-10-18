# Recorded TORAX evidence

These three completed local CPU TORAX datasets let the research agent compare
three different controller changes using saved evidence. Reading them does not
start training or call a provider. Each dataset uses the same SPARC-like setup,
development seed 29 and one round. They are not three different reactors or
independent training-seed replications, and seed 29 is not confirmation data.

| Dataset | Changed knob | Baseline → candidate | Recorded comparison |
| --- | --- | --- | --- |
| SAC actor learning | `sac_actor_lr` | 0.0003 → 0.0001 | [dc0df0a8](shadow/dc0df0a81413479e979a18895e1fe2fe/comparison.json) |
| SAC critic mixing | `sac_mix` | 0.25 → 0.1 | [fbb83d8e](shadow/fbb83d8ee93a424497aeacc0a2633dc3/comparison.json) |
| TD3 action coordinates | `heating_critic_coordinates` | `normalized` → `physical` | [12eddceb](shadow/12eddcebd2c548d889b41d3f2adace91/comparison.json) |

Each comparison contains baseline and candidate results for local, reset-only,
uniform and clipped controls. Each executed 56 episodes: 32 training and
24 evaluation, with 1,680 action windows. Across the three datasets this totals
168 episodes, 96 training, 72 evaluation and 5,040 windows. The initial, gate
and report evaluations are included in evaluation totals. Earlier calibration
and replay-only optimizer work are additional costs.

The [three-dataset provenance index](3demo-datasets.json) records exact
configurations, recorded recommendations, per-file SHA-256 hashes, executed
costs and the source-verification scope. For every bundle, `comparison.json`,
`manifest.json`, `config.json`, `source.zip` and `output/result.json` are
byte-identical to the completed trial. Only `run.json` is projected to the
five fields needed by the evidence adapter: `id`, `status`, `config`, `manifest`
and `comparison`. Process and event metadata are omitted. The index preserves
the original run-file digest and the exported file digest separately.

All 57 ZIP members per dataset match both their frozen manifest hashes and
already-public Git source bytes at commit
`2837fac587a87a07a731d8de199ef32db6df7eb6`. That commit identifies matching source
bytes; it is not claimed as a recorded launch commit. The catalogue verifies
the six-file bundle and recomputes comparison arithmetic before exposing it.
It does not execute the archived code or perform a fresh trajectory/physics
audit. No provider responses, web snapshots, credentials or runtime logs are
included in these exports.

These are completed simulation datasets, not validated efficacy claims. Earlier
research-agent reports on them omitted required facts or failed; exporting the
underlying evidence does not retroactively mark those reports as passes.
Endpoint cost is not shots-to-competence, and a candidate beating its baseline
within an arm is not by itself proof that federation beats local learning.
Keep negative, null and inconclusive results visible. Never edit a recorded
result to fit a preferred explanation or newer code.

## Earlier seed-17 starter evidence

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
