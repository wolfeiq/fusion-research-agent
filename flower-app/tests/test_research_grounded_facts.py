"""Adversarial deterministic publication tests; no provider or physics calls."""
import hashlib
import json
from copy import deepcopy

import pytest

from fusion_agent.research.evidence import resolve_pointer
from fusion_agent.research.grounded_facts import build_grounded_facts, source_provenance
from fusion_agent.research.shadow_analysis import ARMS


def encode(source, document):
    source["content"] = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)
    source["sha256"] = hashlib.sha256(source["content"].encode()).hexdigest()
    return source


def endpoint(cost, competent=False):
    return {"mean_cost": cost, "competent": competent, "valid_complete": True,
            "cases": 1, "complete_cases": 1, "cases_with_events": 0,
            "cases_with_invalid_physics": 0, "condition_signature": "1" * 64}


@pytest.fixture
def comparison():
    branches = {}
    for name, cost in (("baseline", .2), ("candidate", .1)):
        arms = {arm: {"raw": endpoint(cost), "deployed": endpoint(cost),
                      "gate": None if arm == "local" else {"accept": True}}
                for arm in ARMS}
        branches[name] = {"status": "complete", "accounting": {
            "training_episodes": 16, "evaluation_episodes": 12, "windows": 840},
            "configs": {"clipping_initial_norm_fraction": .1 if name == "baseline" else .001},
            "seeds": {"17": {"seed": 17, "status": "complete", "conditions_match": True,
                              "arms": arms, "competence_checkpoints": [{"round": 1}]}}}
    return {"schema": "fusion-shadow-comparison-v1", "status": "complete",
            "execution_status": "completed", "worker_reported_status": "complete",
            "failures": [], "expected_training_seeds": [17], "branches": branches,
            "provenance": {"trial_id": "trial17", "result_sha256": "2" * 64},
            "candidate_vs_baseline_raw_endpoint": {
                arm: {"by_seed": [{"seed": 17, "comparable": True,
                                   "mean_cost_difference": -.1,
                                   "relative_cost_change": -.5,
                                   "baseline_competent": False, "candidate_competent": False}]}
                for arm in ARMS}}


def diagnostic():
    return {"schema": "fusion-shadow-clipping-diagnostic-v1", "status": "available",
            "raw_result_sha256": "2" * 64, "records": [{
                "configuration": "candidate", "seed": 17, "round": 1,
                "role": "deposition_sac", "arm": "clipped",
                "status": "recorded_scalars_consistent", "source_ids": ["donor", "target"],
                "source_count": 2, "thresholds": {"q0": .01, "q1": .01},
                "source_norms": {"donor": {"q0": .02, "q1": .04}, "target": {"q0": .005, "q1": .005}},
                "clipping_factors": {"donor": {"q0": .5, "q1": .25}, "target": {"q0": 1., "q1": 1.}},
                "clipped_sources": {"q0": ["donor"], "q1": ["donor"]},
                "clipped_source_counts": {"q0": 1, "q1": 1},
                "proposal_hashes": {"uniform": "3" * 64, "clipped": "4" * 64},
                "recorded_uniform_and_clipped_hashes_equal": False}]}


@pytest.fixture
def source(comparison):
    comparison["round_aggregation_diagnostics"] = diagnostic()
    return encode({"id": "E001", "kind": "experiment", "metadata": {
        "adapter": "shadow_comparison", "verification_scope": "Summary arithmetic only."}}, comparison)


def mutate(source, fn):
    document = json.loads(source["content"])
    fn(document)
    return encode(source, document)


def kinds(source, kind):
    return [card for card in build_grounded_facts(source) if card["kind"] == kind]


def test_all_cards_are_exact_source_bound_atomic_projections(source):
    original = deepcopy(source)
    cards = build_grounded_facts([source])
    assert cards and source == original
    assert {card["fact_type"] for card in cards} == {
        "trial_status", "actual_budget", "configuration", "paired_raw_endpoint",
        "paired_competence", "gate_outcome", "clipping"}
    for card in cards:
        assert card["source_sha256"] == source["sha256"]
        assert card["provenance"]["origin"] == "saved_shadow_snapshot"
        for citation in card["evidence"]:
            assert citation["source_id"] == source["id"]
            actual = resolve_pointer(source["content"], citation["locator"])
            assert json.loads(citation["quote"]) == actual
    assert "56 total" in kinds(source, "actual_budget")[0]["text"]
    assert "1680 action windows" in kinds(source, "actual_budget")[0]["text"]
    assert all("%" not in card["text"] for card in cards)
    clip = kinds(source, "clipping")
    assert len(clip) == 2
    assert "candidate, training seed 17, round 1" in clip[0]["text"]
    assert "deposition SAC q0" in clip[0]["text"]
    assert "1 of 2" in clip[0]["text"]


def test_registry_list_duplicate_and_dict_content_have_stable_ids(source):
    expected = build_grounded_facts(source)
    assert build_grounded_facts({source["id"]: source}) == expected
    assert build_grounded_facts([source, deepcopy(source)]) == expected
    dictionary = deepcopy(source)
    dictionary["content"] = json.loads(dictionary["content"])
    assert build_grounded_facts(dictionary) == expected
    alias = deepcopy(source)
    alias["canonicalcontent"] = alias.pop("content")
    assert build_grounded_facts(alias) == expected


def test_hash_or_source_identity_changes_cards(source):
    changed = deepcopy(source)
    changed["id"] = "E002"
    assert {card["id"] for card in build_grounded_facts(source)}.isdisjoint(
        card["id"] for card in build_grounded_facts(changed))
    changed["id"] = source["id"]
    mutate(changed, lambda d: d.update(failures=["failure"]))
    assert build_grounded_facts([source, changed]) == []


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64), ("metadata", None), ("kind", "web"), ("id", "../../unsafe")])
def test_missing_or_bad_source_binding_rejected(source, field, value):
    source[field] = value
    assert build_grounded_facts(source) == []
    assert source_provenance(source) is None


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema="unknown"),
    lambda d: d.update(status=True),
    lambda d: d.update(provenance=[]),
])
def test_malformed_or_unknown_source_fails_closed(source, change):
    mutate(source, change)
    assert build_grounded_facts(source) == []


@pytest.mark.parametrize("change", [
    lambda d: d.update(status="failed", execution_status="failed", worker_reported_status="failed"),
    lambda d: d.update(status="partial"),
    lambda d: d.update(failures=[{"seed": 17, "error": "worker stopped"}]),
    lambda d: d["branches"]["candidate"].update(status="failed"),
    lambda d: d["branches"]["candidate"]["seeds"]["17"].update(conditions_match=False),
    lambda d: d["branches"]["candidate"]["seeds"]["17"]["arms"].pop("local"),
])
def test_failed_or_false_complete_not_inactive_clipping_or_positive_comparison(source, change):
    mutate(source, change)
    assert {card["kind"] for card in build_grounded_facts(source)} <= {"trial_status"}


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="invalid_recorded_telemetry"),
    lambda r: r.update(seed=43),
    lambda r: r.update(round=2),
    lambda r: r.update(role="made_up_role"),
    lambda r: r.update(arm="uniform"),
    lambda r: r.update(source_count=3),
    lambda r: r["clipping_factors"]["donor"].update(q0=.9),
    lambda r: r["clipped_source_counts"].update(q0=0),
    lambda r: r["clipped_sources"].update(q0=[]),
    lambda r: r["proposal_hashes"].pop("clipped"),
    lambda r: r.pop("source_norms"),
])
def test_incomplete_or_inconsistent_telemetry_never_publishes_clipping(source, change):
    mutate(source, lambda d: change(d["round_aggregation_diagnostics"]["records"][0]))
    assert not kinds(source, "clipping")


def test_telemetry_other_result_digest_not_published(source):
    mutate(source, lambda d: d["round_aggregation_diagnostics"].update(raw_result_sha256="9" * 64))
    assert not kinds(source, "clipping")


def test_partial_diagnostics_do_not_claim_unrecorded_role(source):
    mutate(source, lambda d: d["round_aggregation_diagnostics"].update(status="partial_or_unavailable"))
    cards = kinds(source, "clipping")
    assert len(cards) == 2 and all(card["context"]["role"] == "deposition_sac" for card in cards)
    assert "heating" not in " ".join(card["text"] for card in cards)


def test_duplicate_telemetry_identity_and_paired_seed_fail_closed(source):
    def change(d):
        records = d["round_aggregation_diagnostics"]["records"]
        records.append(deepcopy(records[0]))
        rows = d["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"]
        rows.append(deepcopy(rows[0]))
    mutate(source, change)
    assert not kinds(source, "clipping")
    assert all(card["context"]["arm"] != "clipped" for card in kinds(source, "paired_raw_endpoint"))


def test_zero_clipping_requires_consistent_positive_norm_evidence(source):
    def change(d):
        row = d["round_aggregation_diagnostics"]["records"][0]
        row["thresholds"] = {"q0": 1., "q1": 1.}
        row["clipping_factors"] = {name: {"q0": 1., "q1": 1.} for name in row["source_ids"]}
        row["clipped_sources"] = {"q0": [], "q1": []}
        row["clipped_source_counts"] = {"q0": 0, "q1": 0}
    mutate(source, change)
    assert all("0 of 2" in card["text"] for card in kinds(source, "clipping"))
    mutate(source, lambda d: d["round_aggregation_diagnostics"]["records"][0].pop("source_norms"))
    assert not kinds(source, "clipping")


def test_gate_fallback_cannot_become_raw_improvement(source):
    def change(d):
        row = d["branches"]["candidate"]["seeds"]["17"]["arms"]["clipped"]
        row.update(raw=endpoint(.3), deployed=endpoint(.15), gate={"accept": False})
        pair = d["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0]
        pair["mean_cost_difference"] = .3 - .2
        pair["deployed_comparison"] = {"mean_cost_difference": -.05}
    mutate(source, change)
    pair = next(card for card in kinds(source, "paired_raw_endpoint") if card["context"]["arm"] == "clipped")
    assert "0.09999999999999998" in pair["text"]
    assert "-0.05" not in pair["text"]
    gate = next(card for card in kinds(source, "gate_outcome")
                if card["context"] == {"configuration": "candidate", "seed": 17,
                                       "arm": "clipped", "outcome": "gate_selected_endpoint"})
    assert "rejected" in gate["text"] and "raw endpoint mean cost is 0.3" in gate["text"]
    assert "deployed endpoint mean cost is 0.15" in gate["text"]


@pytest.mark.parametrize("change", [
    lambda d: d["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0].update(mean_cost_difference=-999),
    lambda d: d["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0].update(candidate_competent=True),
    lambda d: d["branches"]["candidate"]["seeds"]["17"]["arms"]["clipped"]["raw"].update(condition_signature="9" * 64),
    lambda d: d["branches"]["candidate"]["seeds"]["17"]["arms"]["clipped"]["raw"].update(valid_complete=False),
])
def test_invalid_paired_metrics_or_competence_not_published(source, change):
    mutate(source, change)
    assert all(card["context"]["arm"] != "clipped" for card in kinds(source, "paired_raw_endpoint"))
    assert all(card["context"]["arm"] != "clipped" for card in kinds(source, "paired_competence"))


def test_actual_accounting_never_uses_planned_budget(source):
    mutate(source, lambda d: (d.update(frozen_budget={"total_episodes": 999}),
                             d["branches"]["candidate"]["accounting"].pop("training_episodes")))
    assert not kinds(source, "actual_budget")


def test_gate_acceptance_cannot_contradict_its_selected_endpoint(source):
    mutate(source, lambda d: d["branches"]["candidate"]["seeds"]["17"]["arms"]["clipped"].update(deployed=endpoint(.01)))
    assert not any(card["context"]["configuration"] == "candidate" and card["context"]["arm"] == "clipped"
                   for card in kinds(source, "gate_outcome"))


def test_tiny_incorrect_arithmetic_is_not_accepted_as_tolerance(source):
    mutate(source, lambda d: d["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0].update(mean_cost_difference=-.1 + 1e-13))
    assert all(card["context"]["arm"] != "clipped" for card in kinds(source, "paired_raw_endpoint"))


def test_live_wrapper_has_explicit_origin_and_nested_pointers(comparison):
    wrapper = {"schema": "fusion-advisor-shadow-evidence-v1", "status": "completed",
               "comparison": comparison, "policy_activated": False,
               "trial_id": "live-trial17", "round_aggregation_diagnostics": diagnostic()}
    # Deliberately E-prefixed: identity prefixes do not decide origin.
    source = encode({"id": "E001", "kind": "experiment",
                     "metadata": {"adapter": "shadow_execution", "input_source_id": "Dsource"}}, wrapper)
    cards = build_grounded_facts(source)
    assert cards and all(card["provenance"]["origin"] == "research_shadow_execution" for card in cards)
    assert source_provenance(source)["trial_id"] == "live-trial17"
    assert any(card["kind"] == "policy_activation" for card in cards)
    pairs = kinds(source, "paired_raw_endpoint")
    assert all(e["locator"].startswith("/comparison/") for card in pairs for e in card["evidence"])
    mutate(source, lambda d: d.update(status="failed"))
    assert {card["kind"] for card in build_grounded_facts(source)} == {"trial_status", "policy_activation"}


def test_nonfinite_and_duplicate_json_keys_rejected(source):
    for content in ('{"schema":"fusion-shadow-comparison-v1","bad":NaN}',
                    '{"schema":"fusion-shadow-comparison-v1","status":"complete","status":"failed"}'):
        source["content"] = content
        source["sha256"] = hashlib.sha256(content.encode()).hexdigest()
        assert build_grounded_facts(source) == []


@pytest.mark.parametrize("status", ["failed", "timed_out", "interrupted"])
def test_worker_abort_before_result_still_has_current_failure_evidence(status):
    source = encode({"id": "Dfailed", "kind": "experiment", "metadata": {"adapter": "shadow_execution"}},
                    {"schema": "fusion-advisor-shadow-evidence-v1", "status": status,
                     "comparison": None, "policy_activated": False, "trial_id": "aborted17"})
    cards = build_grounded_facts(source)
    assert {card["kind"] for card in cards} == {"trial_status", "policy_activation"}
    status_card = next(card for card in cards if card["kind"] == "trial_status")
    assert f"status {status}" in status_card["text"]
    assert status_card["provenance"]["origin"] == "research_shadow_execution"
    assert status_card["provenance"]["trial_id"] == "aborted17"
    assert status_card["evidence"] == [
        {"source_id": "Dfailed", "locator": "/status", "quote": json.dumps(status)},
        {"source_id": "Dfailed", "locator": "/comparison", "quote": "null"}]


def test_completed_wrapper_without_comparison_cannot_become_success_evidence():
    source = encode({"id": "Dinvalid", "kind": "experiment", "metadata": {"adapter": "shadow_execution"}},
                    {"schema": "fusion-advisor-shadow-evidence-v1", "status": "completed",
                     "comparison": None, "policy_activated": False})
    assert build_grounded_facts(source) == []
