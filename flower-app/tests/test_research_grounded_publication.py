"""Adversarial publication checks, with synthetic records and no provider calls."""

import copy
import hashlib
import inspect
import json

import pytest

from fusion_agent.research.grounded_facts import build_grounded_facts
from fusion_agent.research.grounded_publication import build_publication
from fusion_agent.research.shadow_analysis import _next_run_proposal


def captured(identifier, content, *, kind="experiment", adapter="shadow_comparison", **metadata):
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
    return {"id": identifier, "content": text, "kind": kind,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "uri": "https://example.org/paper" if kind == "web" else "saved:trial",
            "title": "Captured source", "metadata": {"adapter": adapter, **metadata}}


def comparison():
    def endpoint(cost):
        return {"mean_cost": cost, "competent": False, "valid_complete": True,
                "cases": 1, "complete_cases": 1, "cases_with_events": 0,
                "cases_with_invalid_physics": 0, "condition_signature": "a" * 64}
    branches = {}
    for name, cost in (("baseline", 1.0), ("candidate", .8)):
        branches[name] = {
            "status": "complete", "configs": {"clipping_initial_norm_fraction": .1 if name == "baseline" else .001},
            "accounting": {"training_episodes": 16, "evaluation_episodes": 12, "windows": 840},
            "seeds": {"17": {"seed": 17, "status": "complete", "conditions_match": True,
                               "arms": {arm: {"raw": endpoint(cost), "deployed": endpoint(cost), "gate": None}
                                        for arm in ("local", "reset_only", "uniform", "clipped")}}},
        }
    recommendation = {"code": "repeat_varied_paired_seeds", "automatic_apply": False,
                      "automatic_launch": False, "efficacy_claim": False,
                      "text": "Untrusted prose must not replace the server template."}
    result = {
        "schema": "fusion-shadow-comparison-v1", "status": "complete", "execution_status": "completed",
        "worker_reported_status": "complete", "failures": [], "expected_training_seeds": [17],
        "branches": branches, "candidate_vs_baseline_raw_endpoint": {
            "clipped": {"by_seed": [{"seed": 17, "comparable": True,
                                     "mean_cost_difference": .8 - 1.0,
                                     "baseline_competent": False, "candidate_competent": False}]}},
        "recommendation": recommendation,
    }
    result["next_run_proposal"] = _next_run_proposal(
        {"change": "clipping_initial_norm_fraction", "value": .001, "seeds": [17], "rounds": 1},
        True, recommendation,
    )
    return result


def sources():
    return [captured("E001", comparison()),
            captured("W001", "Clipping limits updates in this stated setting. Other settings need evaluation.",
                     kind="web", status="extracted")]


def requests():
    return [{"source_id": "W001", "anchor": "Clipping limits", "max_words": 8}]


def selected(source_list, kind="paired_raw_endpoint"):
    return [card["id"] for card in build_grounded_facts(source_list) if card["kind"] == kind]


def test_publication_only_accepts_source_selections_not_claim_prose():
    assert set(inspect.signature(build_publication).parameters) == {
        "sources", "fact_ids", "literature_requests", "next_run_source_id"}
    data = sources()
    original = copy.deepcopy(data)
    report = build_publication(data, selected(data), requests(), "E001")
    assert data == original
    assert report["status"] == "grounded"
    assert report["scientific_efficacy_established"] is False
    assert report["automatic_policy_changes_allowed"] is False
    assert repr(.8 - 1.0) in report["answer"]
    assert "baseline mean cost is 1.0" not in report["answer"]
    assert "candidate mean cost is 0.8" not in report["answer"]
    assert "Untrusted prose" not in report["answer"]
    assert report["facts"][0]["provenance"]["origin"] == "saved_shadow_snapshot"
    assert report["facts"][0]["source_label"] == "Saved shadow trial"
    assert "Saved shadow trial:" in report["answer"]
    assert report["literature_excerpts"][0]["quote"] == "Clipping limits updates in this stated setting. Other"
    proposal = report["recommendation"]
    assert proposal["budget"]["total_episodes"] == 112
    assert proposal["budget"]["training_episodes"] + proposal["budget"]["evaluation_episodes"] == 112
    assert "subsets" in proposal["budget_scope"]
    assert proposal["request"] == comparison()["next_run_proposal"]["request"]


def test_exact_quote_cannot_be_promoted_to_broader_literature_assertion():
    data = sources()
    data[1] = captured("W001", "Centralized DP-SGD clips per-example gradients. Client-level methods are different.", kind="web", status="extracted")
    report = build_publication(data, selected(data), [{"source_id": "W001", "anchor": "Centralized DP-SGD", "max_words": 5}])
    assert "Source excerpt" in report["answer"]
    assert "Centralized DP-SGD clips per-example gradients." in report["answer"]
    assert "guarantees client-level" not in report["answer"]
    assert report["literature_excerpts"][0]["publication_basis"] == "exact_attributed_excerpt"


def test_current_execution_origin_is_metadata_not_id_prefix():
    data = sources()
    data[0] = captured("E001", {"schema": "fusion-advisor-shadow-evidence-v1", "status": "completed",
                                 "comparison": comparison(), "policy_activated": False}, adapter="shadow_execution")
    report = build_publication(data, selected(data), requests(), "E001")
    assert report["facts"][0]["provenance"]["origin"] == "research_shadow_execution"
    assert "Current research-run shadow trial:" in report["answer"]
    assert report["recommendation"]["evidence"][0]["locator"].startswith("/comparison/")


@pytest.mark.parametrize("mutation", [
    lambda data: data[0].update(sha256="0" * 64),
    lambda data: data[1].pop("sha256"),
    lambda data: data.append(copy.deepcopy(data[0])),
    lambda data: data[0].update(kind="model_claim"),
])
def test_unbound_unknown_or_duplicate_sources_fail_closed(mutation):
    data = sources()
    ids = selected(data)
    mutation(data)
    with pytest.raises(ValueError):
        build_publication(data, ids, requests())


def test_unknown_and_duplicate_fact_ids_rejected():
    data = sources()
    with pytest.raises(ValueError, match="Unknown grounded"):
        build_publication(data, ["Finvented"], requests())
    ids = selected(data)
    with pytest.raises(ValueError, match="Duplicate grounded"):
        build_publication(data, ids + ids, requests())


@pytest.mark.parametrize("excerpt_request,match", [
    ({"source_id": "W999", "anchor": "Clipping", "max_words": 5}, "Unknown literature"),
    ({"source_id": "E001", "anchor": "Clipping", "max_words": 5}, "extracted"),
    ({"source_id": "W001", "anchor": "Invented words", "max_words": 5}, "not found"),
    ({"source_id": "W001", "anchor": "Clipping", "max_words": 26}, "1 to 25"),
    ({"source_id": "W001", "anchor": "Clipping", "max_words": 5, "text": "Claim"}, "exactly"),
])
def test_invalid_literature_request_cannot_publish(excerpt_request, match):
    data = sources()
    with pytest.raises(ValueError, match=match):
        build_publication(data, selected(data), [excerpt_request])


@pytest.mark.parametrize("text,anchor,max_words,match", [
    ("Repeated words here. Repeated words there.", "Repeated words", 3, "Ambiguous"),
    ("Clean anchor bad\x1a extraction.", "Clean anchor", 4, "control characters"),
])
def test_ambiguous_and_corrupt_pdf_excerpts_are_rejected(text, anchor, max_words, match):
    data = sources()
    data[1] = captured("W001", text, kind="web", status="extracted")
    with pytest.raises(ValueError, match=match):
        build_publication(data, selected(data), [{"source_id": "W001", "anchor": anchor, "max_words": max_words}])


def test_explicit_body_locator_revalidates_helper_candidate_and_cannot_select_bibliography():
    from fusion_agent.research.web_citations import cite_web_excerpt

    sentence = "The actor maximizes expected reward together with policy entropy."
    data = sources()
    data[1] = captured("W001", "Abstract\n" + sentence + "\nBibTeX\nabstract = { " + sentence + " }",
                       kind="web", status="extracted")
    args = {"source_id": "W001", "anchor": "The actor maximizes expected reward together", "max_words": 9}
    choices = cite_web_excerpt(data[1], args["anchor"], args["max_words"])["candidates"]
    original = copy.deepcopy(data)
    report = build_publication(data, selected(data), [{**args, "locator": choices[0]["citation"]["locator"]}])
    assert report["literature_excerpts"][0]["evidence"] == [choices[0]["citation"]]
    assert report["literature_excerpts"][0]["quote"] == sentence
    assert data == original
    for locator in ["chars:0:10", "chars:999:9999", None, "made-up"]:
        with pytest.raises(ValueError, match="locator"):
            build_publication(data, selected(data), [{**args, "locator": locator}])
    with pytest.raises(ValueError, match="outside retained"):
        build_publication(data, selected(data), [{**args, "locator": choices[1]["citation"]["locator"]}])


def test_explicit_locator_cannot_resolve_duplicate_scientific_body_sentences():
    from fusion_agent.research.web_citations import cite_web_excerpt

    sentence = "The actor maximizes expected reward together with policy entropy."
    data = sources()
    data[1] = captured("W001", "Abstract\n" + sentence + " " + sentence, kind="web", status="extracted")
    args = {"source_id": "W001", "anchor": "The actor maximizes", "max_words": 9}
    locator = cite_web_excerpt(data[1], args["anchor"], args["max_words"])["candidates"][0]["citation"]["locator"]
    with pytest.raises(ValueError, match="Ambiguous"):
        build_publication(data, selected(data), [{**args, "locator": locator}])


def test_quote_budget_applies_across_ids_for_same_url_and_duplicate_requests():
    data = sources()
    data[1] = captured("W001", " ".join(f"word{i}" for i in range(40)), kind="web", status="extracted")
    data.append({**data[1], "id": "W002"})
    with pytest.raises(ValueError, match="exceed 25"):
        build_publication(data, selected(data), [
            {"source_id": "W001", "anchor": "word0", "max_words": 20},
            {"source_id": "W002", "anchor": "word20", "max_words": 10},
        ])
    with pytest.raises(ValueError, match="Duplicate literature"):
        build_publication(sources(), selected(sources()), requests() * 2)


@pytest.mark.parametrize("field,value", [("total_episodes", 136), ("training_episodes", True), ("gate_episodes", 8)])
def test_budget_miscalculation_rejected_even_if_source_hash_is_valid(field, value):
    record = comparison()
    record["next_run_proposal"]["budget"][field] = value
    data = [captured("E001", record), sources()[1]]
    with pytest.raises(ValueError, match="budget"):
        build_publication(data, selected(data), requests(), "E001")


@pytest.mark.parametrize("mutation", [
    lambda record: record["recommendation"].update(code="deploy_winner"),
    lambda record: record["next_run_proposal"].update(automatic_apply=True),
    lambda record: record["recommendation"].update(efficacy_claim=True),
    lambda record: record["next_run_proposal"]["request"].update(value=1000),
    lambda record: record.update(execution_status="failed"),
])
def test_unrecognized_or_unsafe_proposals_never_published(mutation):
    record = comparison()
    mutation(record)
    data = [captured("E001", record), sources()[1]]
    with pytest.raises(ValueError):
        build_publication(data, selected(data, "trial_status"), requests(), "E001")


def test_absent_data_stays_insufficient_instead_of_fabricating_a_fact():
    data = sources()
    assert build_publication(data, [], requests())["status"] == "insufficient_evidence"
    assert build_publication(data, selected(data), [])["status"] == "insufficient_evidence"


def test_partial_record_does_not_publish_successful_comparison():
    record = comparison()
    record["status"] = "partial"
    record["execution_status"] = "timed_out"
    data = [captured("E001", record), sources()[1]]
    assert selected(data) == []
    report = build_publication(data, selected(data, "trial_status"), requests())
    assert "timed_out" in report["answer"]
    assert "-0.2" not in report["answer"]


def test_mixed_saved_measurements_and_current_failed_status_are_explicitly_distinct():
    data = sources()
    document = {"schema": "fusion-advisor-shadow-evidence-v1", "status": "failed",
                "trial_id": "failed-current", "comparison": None, "policy_activated": False}
    data.append(captured("D001", document, adapter="shadow_execution"))
    ids = selected(data) + [card["id"] for card in build_grounded_facts(data)
                            if card["source_id"] == "D001" and card["kind"] == "trial_status"]
    report = build_publication(data, ids, requests(), "D001")
    assert "Saved shadow trial:" in report["answer"]
    assert "Current research-run shadow trial failed-current:" in report["answer"]
    assert report["recommendation"]["code"] == "diagnose_before_comparison"
    assert report["recommendation"]["request"] is None
    assert report["recommendation"]["budget"] is None
    assert report["recommendation"]["evidence"][0]["locator"] == "/status"
