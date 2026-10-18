# Verified saved-data demo

This configuration completed on deployed code `94ad4ba` as run
`92753c1d8f4b49d5b3e62296d98536dc`, passing all 44 reporting checks. A future live
request still depends on model/search availability and output quality.

1. Select only **TD3 critic action coordinates: normalized → physical**.
2. Leave **Live TORAX generation** unchecked (experimental beta).
3. Paste the question below and choose **Analyze saved data + literature**.
4. Inspect measurements, the recommendation, trace and source links. The result
   should distinguish saved development evidence from a newly executed trial.

For a provider-independent fallback choose **Inspect local evidence** with that
same one dataset. It displays saved measurements without an AI agent, web search
or new training. It is explicitly labelled local inspection.

## Exact question used in verification

```text
Audit only the selected published SPARC-like development seed-29, one-round TD3 critic-action-coordinate pilot. Do not launch training, shadow work or modify any experiment. Label these as saved results, not a new trial. Report the saved trial status, the exact normalized-to-physical critic coordinate change, actual episode accounting, all four candidate-minus-baseline RAW endpoint cost differences (local, reset_only, uniform, clipped), and all four baseline/candidate competence pairs. Prioritize those 11 fact cards; include the candidate SAC clipping diagnostic if the 12-card limit permits. Preserve trade-offs and distinguish raw learning from gate fallback. Decide whether to keep the baseline and expose the recorded conservative next-run proposal and its full training/evaluation budget without launching it. Use Tavily to find and extract a primary publication on TD3 critic learning or fair multi-seed RL evaluation. Select one short exact scientific-body excerpt relevant to that methodological issue, not authors, proceedings, bibliography or page metadata. Explain that a single development seed, endpoint cost and a paper quotation do not prove cold-start acceleration. Keep any untested mechanism or hypothesis separate from recorded facts.
```

## Scope

The dataset is one SPARC-like development seed, one round, 56 executed episodes
(32 training, 24 evaluation). No arm reached competence. A lower endpoint cost
does not establish faster shots-to-competence. The saved recommendation is to
keep the baseline; its proposed 112-episode follow-up is not launched by analysis.
The cited TD3 paper supplies critic-methodology context, not causal proof of this
change or evidence that federation solves cold start.
