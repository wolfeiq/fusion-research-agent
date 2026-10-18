"""The real graph selects evidence; even a gullible judge cannot author facts."""

import copy
import hashlib
import json

import pytest
from test_research_graph import FakeRetrieval, TeamModel
from test_research_grounded_facts import encode

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.grounded_excerpts import build_excerpt_catalogue
from fusion_agent.research.grounded_facts import build_grounded_facts
from fusion_agent.research.grounded_writer import (
    _draft,
    finalize_grounded,
    review_grounded,
)

pytest_plugins = ["test_research_grounded_facts"]


@pytest.fixture
def source(source):
    """The saved reporting fixture includes the real recorded proposal contract."""
    from fusion_agent.research.shadow_analysis import _next_run_proposal

    document = json.loads(source["content"])
    recommendation = {"code": "keep_baseline", "automatic_apply": False,
                      "automatic_launch": False, "efficacy_claim": False}
    document["recommendation"] = recommendation
    document["next_run_proposal"] = _next_run_proposal(
        {"change": "clipping_initial_norm_fraction", "value": .001, "seeds": [17], "rounds": 1},
        True, recommendation,
    )
    return encode(source, document)


def web_source():
    content = "In centralized training, gradients are clipped for sample-level privacy."
    return {"id": "W002", "kind": "web", "title": "Synthetic source", "uri": "https://proceedings.mlr.press/v80/fujimoto18a.html",
            "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "metadata": {"status": "extracted"}}


def engine_for(source, model=None):
    engine = ResearchEngine([source, web_source()], model, None, lambda event: None)
    engine.read_ids.add(source["id"])
    return engine


def selection(source):
    cards = build_grounded_facts([source])
    return {"fact_ids": [next(c["id"] for c in cards if c["kind"] == "paired_raw_endpoint")],
            "excerpt_ids": [build_excerpt_catalogue([web_source()])[0]["id"]],
            "hypotheses": [{"text": "The absolute cost is 999 and this establishes client-level privacy.",
                            "evidence_ids": [source["id"], "W002"]}],
            "next_run_source_id": "", "gaps": []}


def call(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "bounded", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class ApprovingJudge:
    def complete(self, messages, tools, tool_choice="auto"):
        return call("review_evidence_selection", {"decision": "accept", "gaps": [],
                    "rejected_hypothesis_indices": []}), {}


def test_bad_model_interpretation_never_gains_authority_even_when_judge_approves(source):
    engine = engine_for(source, ApprovingJudge())
    draft = _draft(engine, selection(source))
    reviewed = review_grounded(engine, {"draft": draft, "question": "What does the experiment establish?", "round": 1})["report"]
    assert reviewed["evidence_sufficient"]
    assert "999" not in reviewed["answer"]
    assert "client-level privacy" not in reviewed["answer"]
    assert "In centralized training" in reviewed["answer"]
    assert reviewed["model_interpretation"]["hypotheses"][0]["status"] == "untested"
    assert reviewed["grounded_publication"]["scientific_efficacy_established"] is False
    assert reviewed["grounded_publication"]["automatic_policy_changes_allowed"] is False
    assert all("999" not in claim["text"] for claim in reviewed["claims"])


def test_final_boundary_recompiles_instead_of_trusting_changed_report_text(source):
    engine = engine_for(source)
    draft = _draft(engine, selection(source))
    expected = copy.deepcopy(draft["grounded_publication"])
    draft.update(answer="Federation always wins: cost999", claims=[{"text": "Made up"}],
                 limitations=["No trajectories exist"], stop_reason="All claims verified", judge={"decision": "accept"})
    draft["grounded_publication"]["facts"][0]["text"] = "tampered"
    result = finalize_grounded(engine, draft)
    assert result["grounded_publication"] == expected
    assert result["answer"] == expected["answer"]
    assert result["claims"] == expected["facts"]
    assert "999" not in result["answer"]
    assert "No trajectories exist" not in result["limitations"]


@pytest.mark.parametrize("field,value", [("fact_ids", ["Fimaginary"]), ("next_run_source_id", {})])
def test_selection_is_typed_and_cannot_add_evidence(source, field, value):
    args = selection(source)
    args[field] = value
    with pytest.raises((ValueError, TypeError)):
        _draft(engine_for(source), args)


def test_fact_text_and_literature_paraphrases_are_not_accepted_as_inputs(source):
    args = selection(source)
    args["claims"] = [{"text": "Federation cures cold start"}]
    with pytest.raises(ValueError, match="exactly"):
        _draft(engine_for(source), args)
    args = selection(source)
    args["excerpt_ids"] = [{"quote": "This supports client-level privacy"}]
    with pytest.raises(ValueError):
        _draft(engine_for(source), args)


def test_final_boundary_detects_source_mutation(source):
    engine = engine_for(source)
    draft = _draft(engine, selection(source))
    engine.sources[source["id"]]["content"] += " "
    with pytest.raises(ValueError, match="hash"):
        finalize_grounded(engine, draft)


@pytest.mark.parametrize("value", [["Qimaginary"], ["same", "same"], "[]"])
def test_excerpt_selection_cannot_invent_or_duplicate_quotations(source, value):
    args = selection(source)
    args["excerpt_ids"] = value
    with pytest.raises(ValueError):
        _draft(engine_for(source), args)


def test_final_boundary_reuses_server_question_for_ranked_catalogue(source, monkeypatch):
    from fusion_agent.research import grounded_writer

    original = build_excerpt_catalogue
    questions = []

    def catalogue(sources, question=""):
        questions.append(question)
        return original(sources, question)

    monkeypatch.setattr(grounded_writer, "build_excerpt_catalogue", catalogue)
    engine = engine_for(source)
    report = _draft(engine, selection(source), "Does this apply to centralized training?")
    result = finalize_grounded(engine, report)
    assert questions == ["Does this apply to centralized training?"] * 2
    assert result["grounded_publication"]["literature_excerpts"]


def test_optional_hypothesis_unknown_sources_are_quarantined_not_guessed(source):
    engine = engine_for(source, ApprovingJudge())
    args = selection(source)
    args["hypotheses"][0]["evidence_ids"] = [args["fact_ids"][0], "W-imaginary"]
    draft = _draft(engine, args)
    assert draft["model_interpretation"]["hypotheses"] == []
    rejected = draft["model_interpretation"]["rejected_hypotheses"]
    assert len(rejected) == 1
    assert rejected[0]["status"] == "invalid_source_references"
    assert rejected[0]["evidence_ids"] == []
    assert rejected[0]["rejected_evidence_ids"] == args["hypotheses"][0]["evidence_ids"]
    reviewed = review_grounded(engine, {"draft": draft, "question": "What was measured?", "round": 1})["report"]
    result = finalize_grounded(engine, reviewed)
    assert result["evidence_sufficient"]
    assert "999" not in result["answer"]
    assert result["model_interpretation"]["rejected_hypotheses"] == rejected
    args["fact_ids"] = ["F-imaginary"]
    with pytest.raises(ValueError):
        _draft(engine, args)  # Optional prose rejection must never relax factual checks.


def current_source(saved, *, failed=False):
    from fusion_agent.research.shadow_analysis import _next_run_proposal

    content = json.loads(saved["content"])
    recommendation = {"code": "repeat_varied_paired_seeds", "automatic_apply": False,
                      "automatic_launch": False, "efficacy_claim": False}
    content["recommendation"] = recommendation
    content["next_run_proposal"] = _next_run_proposal(
        {"change": "clipping_initial_norm_fraction", "value": .001, "seeds": [17], "rounds": 1},
        True, recommendation,
    )
    wrapper = {"schema": "fusion-advisor-shadow-evidence-v1", "trial_id": "current-trial",
               "status": "failed" if failed else "completed", "policy_activated": False,
               "comparison": None if failed else content,
               "round_aggregation_diagnostics": content["round_aggregation_diagnostics"]}
    text = json.dumps(wrapper)
    return {"id": "D001", "kind": "experiment", "content": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "metadata": {"adapter": "shadow_execution"}}


def test_current_run_cannot_be_substituted_with_historical_measured_values(source):
    engine = engine_for(source)
    current = current_source(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = selection(source)
    draft = _draft(engine, args)
    facts = draft["grounded_publication"]["facts"]
    assert len(facts) == 12 and {card["source_id"] for card in facts} == {current["id"]}
    for kind in ("paired_raw_endpoint", "paired_competence"):
        assert {card["context"]["arm"] for card in facts if card["fact_type"] == kind} == {
            "local", "reset_only", "uniform", "clipped"}
    assert {card["fact_type"] for card in facts} >= {"trial_status", "configuration_change", "actual_budget", "clipping"}
    assert draft["fact_selection"]["omitted_optional_fact_ids"] == args["fact_ids"]
    assert len(draft["fact_selection"]["automatically_included_fact_ids"]) == 12
    assert "Current research-run shadow" in draft["answer"] and "Saved shadow" not in draft["answer"]


def test_optional_policy_flag_cannot_displace_local_competence(source):
    current = current_source(source)
    engine = engine_for(source, ApprovingJudge())
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    cards = build_grounded_facts(current)
    # Reproduce the observed live omission: twelve valid choices included a
    # policy flag but excluded local competence; the judge accepted the packet.
    chosen = [card["id"] for card in cards if (
        card["fact_type"] in {"trial_status", "configuration_change", "actual_budget", "policy_activation", "paired_raw_endpoint"}
        or card["fact_type"] == "paired_competence" and card["context"]["arm"] != "local"
        or card["fact_type"] == "clipping" and card["context"]["critic"] == "q0")]
    assert len(chosen) == 12
    args = {**selection(source), "fact_ids": chosen}
    draft = _draft(engine, args)
    facts = draft["grounded_publication"]["facts"]
    local = next(card for card in cards if card["fact_type"] == "paired_competence" and card["context"]["arm"] == "local")
    flag = next(card for card in cards if card["fact_type"] == "policy_activation")
    assert local["id"] in {card["id"] for card in facts}
    assert flag["id"] not in {card["id"] for card in facts}
    assert draft["fact_selection"]["automatically_included_fact_ids"] == [local["id"]]
    assert draft["fact_selection"]["omitted_optional_fact_ids"] == [flag["id"]]
    reviewed = review_grounded(engine, {"draft": draft, "question": "All four comparisons", "round": 1})["report"]
    assert reviewed["evidence_sufficient"]
    result = finalize_grounded(engine, reviewed)
    assert result["grounded_publication"]["facts"] == facts
    assert len(result["claims"]) == 12


def test_all_optional_model_choices_cannot_replace_any_current_comparison(source):
    current = current_source(source)
    engine = engine_for(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    saved = build_grounded_facts(source)
    requested = [card["id"] for card in saved[:12]]
    assert len(requested) == 12
    draft = _draft(engine, {**selection(source), "fact_ids": requested})
    assert len(draft["claims"]) == 12
    assert {card["source_id"] for card in draft["claims"]} == {current["id"]}
    assert draft["fact_selection"]["omitted_optional_fact_ids"] == requested
    assert draft["grounded_publication"]["recommendation"]["source_id"] == current["id"]
    assert draft["fact_selection"]["requested_next_run_source_id"] == ""
    assert draft["fact_selection"]["published_next_run_source_id"] == current["id"]
    assert draft["grounded_publication"]["recommendation"]["budget"]["total_episodes"] == 112
    assert draft["grounded_publication"]["recommendation"]["automatic_launch"] is False


@pytest.mark.parametrize("change,missing", [
    (lambda d: d["comparison"]["candidate_vs_baseline_raw_endpoint"]["local"]["by_seed"][0].update(candidate_competent=True), "local paired competence comparison"),
    (lambda d: d["comparison"]["branches"]["candidate"]["accounting"].pop("windows"), "current actual episode accounting"),
    (lambda d: d["comparison"]["branches"]["candidate"]["configs"].update(clipping_initial_norm_fraction=.1), "one recorded baseline-to-candidate configuration change"),
])
def test_missing_current_facet_stays_insufficient_even_with_historical_card_and_judge_acceptance(source, change, missing):
    current = current_source(source)
    document = json.loads(current["content"])
    change(document)
    encode(current, document)
    engine = engine_for(source, ApprovingJudge())
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = selection(source)
    draft = _draft(engine, args)
    coverage = draft["verification"]["current_shadow_coverage"]
    assert coverage["complete"] is False and missing in coverage["missing_requirements"]
    assert draft["grounded_publication"]["status"] == "insufficient_evidence"
    reviewed = review_grounded(engine, {"draft": draft, "question": "All four comparisons", "round": 1})["report"]
    result = finalize_grounded(engine, reviewed)
    assert result["judge"]["decision"] == "accept" and result["evidence_sufficient"] is False
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert missing in result["answer"]
    assert result["grounded_publication"]["recommendation"] is None


def test_missing_current_proposal_is_a_gap_not_an_invented_follow_up(source):
    current = current_source(source)
    document = json.loads(current["content"])
    document["comparison"].pop("next_run_proposal")
    encode(current, document)
    engine = engine_for(source, ApprovingJudge())
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    draft = _draft(engine, selection(source))
    assert not draft["verification"]["current_shadow_coverage"]["complete"]
    assert "recorded current next-run recommendation and proposal" in draft["answer"]
    assert draft["grounded_publication"]["recommendation"] is None


def test_missing_clipping_is_explicit_unavailability_not_inactive_or_missing_core_coverage(source):
    current = current_source(source)
    document = json.loads(current["content"])
    document.pop("round_aggregation_diagnostics")
    encode(current, document)
    engine = engine_for(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = {**selection(source), "fact_ids": []}
    draft = _draft(engine, args)
    assert len(draft["claims"]) == 11
    assert draft["verification"]["current_shadow_coverage"]["complete"]
    assert not any(card["fact_type"] == "clipping" for card in draft["claims"])
    assert any("does not establish inactive clipping" in text for text in draft["limitations"])


@pytest.mark.parametrize("requested", [["Funknown"], ["duplicate", "duplicate"]])
def test_automatic_bundle_never_hides_invalid_optional_selections(source, requested):
    current = current_source(source)
    engine = engine_for(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    with pytest.raises(ValueError):
        _draft(engine, {**selection(source), "fact_ids": requested})


def test_multiple_current_sources_are_not_silently_ranked_or_truncated(source):
    current = current_source(source)
    other = {**copy.deepcopy(current), "id": "D002"}
    engine = engine_for(source)
    engine.sources.update({current["id"]: current, other["id"]: other})
    engine.read_ids.update((current["id"], other["id"]))
    with pytest.raises(ValueError, match="exactly one current shadow"):
        _draft(engine, selection(source))


def test_writer_payload_and_trace_expose_server_included_bundle(source):
    from fusion_agent.research.grounded_writer import write_grounded

    current, events = current_source(source), []

    class EmptyFactSelection:
        def complete(self, messages, tools, tool_choice="auto"):
            payload = json.loads(messages[1]["content"])
            assert len(payload["required_current_fact_ids"]) == 12
            assert payload["optional_fact_slots"] == 0
            assert payload["current_shadow_coverage"]["source_ids"] == [current["id"]]
            return call("submit_evidence_report", {**selection(source), "fact_ids": [], "hypotheses": []}), {}

    engine = ResearchEngine([source, current, web_source()], EmptyFactSelection(), None, events.append)
    engine.read_ids.update((source["id"], current["id"]))
    draft = write_grounded(engine, {"question": "Inspect all four current controls"})["draft"]
    event = next(event for event in events if event["kind"] == "grounded_draft")
    assert event["fact_ids"] == [card["id"] for card in draft["claims"]]
    assert event["fact_selection"]["requested_fact_ids"] == []
    assert len(event["fact_selection"]["automatically_included_fact_ids"]) == 12
    assert event["current_shadow_coverage"]["complete"]


@pytest.mark.parametrize("field,value", [
    ("gaps", "The gap is a string instead of an array"), ("gaps", {}),
    ("gaps", ["same", "same"]), ("gaps", ["x" * 601]),
    ("hypotheses", "No hypothesis"), ("hypotheses", {}),
    ("hypotheses", [{"text": "Missing source IDs"}]),
    ("hypotheses", [{"text": 99, "evidence_ids": []}]),
    ("hypotheses", [{"text": "Untested", "evidence_ids": "D001"}]),
])
def test_current_optional_annotation_errors_are_quarantined_without_losing_verified_facts(source, field, value):
    current = current_source(source)
    engine = engine_for(source, ApprovingJudge())
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = {**selection(source), "fact_ids": [], "hypotheses": [], field: value}
    draft = _draft(engine, args)
    assert len(draft["claims"]) == 12
    annotations = draft["model_interpretation"]["rejected_annotations"]
    assert len(annotations) == 1 and annotations[0]["field"] == field
    assert annotations[0]["status"] == "invalid_optional_annotation"
    assert draft["model_interpretation"][field] == []
    assert any("malformed optional annotations" in text for text in draft["limitations"])
    assert draft["selection"][field] == value  # Preserve the defect in the retained raw selection.
    reviewed = review_grounded(engine, {"draft": draft, "question": "All four controls", "round": 1})["report"]
    final = finalize_grounded(engine, reviewed)
    assert final["evidence_sufficient"]
    assert final["model_interpretation"]["rejected_annotations"] == annotations
    assert final["claims"] == draft["claims"]


def test_current_large_known_selection_is_bounded_after_mandatory_bundle(source):
    current = current_source(source)
    engine = engine_for(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    requested = [card["id"] for card in build_grounded_facts([source, current])]
    assert 40 < len(requested) <= 64
    draft = _draft(engine, {**selection(source), "fact_ids": requested})
    assert len(draft["claims"]) == 12
    assert len(draft["fact_selection"]["omitted_optional_fact_ids"]) == len(requested) - 12
    assert draft["fact_selection"]["requested_fact_ids"] == requested
    assert {card["source_id"] for card in draft["claims"]} == {current["id"]}


@pytest.mark.parametrize("kind", ["unknown_after_twelve", "duplicate", "more_than_sixty_four", "oversized_transport"])
def test_current_quarantine_does_not_relax_factual_or_transport_bounds(source, kind):
    current = current_source(source)
    engine = engine_for(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    ids = [card["id"] for card in build_grounded_facts([source, current])]
    args = {**selection(source), "fact_ids": ids, "gaps": "Malformed optional gaps"}
    if kind == "unknown_after_twelve":
        args["fact_ids"] = ids[:12] + ["Funknown"]
    elif kind == "duplicate":
        args["fact_ids"] = ids[:12] + ids[:1]
    elif kind == "more_than_sixty_four":
        args["fact_ids"] = [f"F{index}" for index in range(65)]
    else:
        args["gaps"] = "x" * 15000
    with pytest.raises(ValueError):
        _draft(engine, args)


def test_multiple_saved_datasets_keep_strict_optional_and_twelve_fact_contract(source):
    engine = engine_for(source)
    other = {**copy.deepcopy(source), "id": "E002"}
    engine.sources[other["id"]] = other
    engine.read_ids.add(other["id"])
    for patch in ({"gaps": "Malformed"}, {"hypotheses": [{}]},
                  {"fact_ids": [card["id"] for card in build_grounded_facts(source)][:13]}):
        with pytest.raises(ValueError):
            _draft(engine, {**selection(source), **patch})


def test_single_saved_trial_restores_omitted_competence_and_proposal_without_new_provenance(source):
    original = copy.deepcopy(source)
    engine = engine_for(source, ApprovingJudge())
    cards = build_grounded_facts(source)
    # The deployed writer omitted clipped competence and supplied no proposal.
    args = {**selection(source), "fact_ids": [card["id"] for card in cards if (
        card["fact_type"] in {"trial_status", "configuration_change", "actual_budget", "paired_raw_endpoint"}
        or card["fact_type"] == "paired_competence" and card["context"]["arm"] != "clipped")],
        "next_run_source_id": ""}
    draft = _draft(engine, args)
    coverage = draft["verification"]["comparison_coverage"]
    assert coverage == draft["verification"]["current_shadow_coverage"]
    assert coverage["scope"] == "single_seed_saved_comparison"
    assert coverage["origin"] == "saved_shadow_snapshot" and coverage["current_source_ids"] == []
    assert coverage["complete"] and len(coverage["required_fact_ids"]) == 12
    for kind in ("paired_raw_endpoint", "paired_competence"):
        assert {card["context"]["arm"] for card in draft["claims"] if card["fact_type"] == kind} == {
            "local", "reset_only", "uniform", "clipped"}
    assert all(card["source_label"] == "Saved shadow trial trial17" for card in draft["claims"])
    assert "Current research-run" not in draft["answer"]
    proposal = draft["grounded_publication"]["recommendation"]
    recorded = json.loads(source["content"])["next_run_proposal"]
    assert proposal["code"] == "keep_baseline" and proposal["request"] == recorded["request"]
    assert proposal["budget"]["total_episodes"] == 112
    assert proposal["automatic_apply"] is proposal["automatic_launch"] is False
    reviewed = review_grounded(engine, {"draft": draft, "question": "All saved controls", "round": 1})["report"]
    assert finalize_grounded(engine, reviewed)["evidence_sufficient"]
    assert source == original


@pytest.mark.parametrize("missing", ["paired_cost", "competence", "configuration", "accounting", "proposal"])
def test_single_saved_trial_cannot_hide_missing_required_evidence_with_judge_acceptance(source, missing):
    document = json.loads(source["content"])
    if missing == "paired_cost":
        document["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0]["mean_cost_difference"] = 999
    elif missing == "competence":
        document["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0].pop("candidate_competent")
    elif missing == "configuration":
        document["branches"]["candidate"]["configs"] = {}
    elif missing == "accounting":
        document["branches"]["candidate"].pop("accounting")
    else:
        document.pop("next_run_proposal")
    encode(source, document)
    engine = engine_for(source, ApprovingJudge())
    args = {"fact_ids": [], "excerpt_ids": [build_excerpt_catalogue([web_source()])[0]["id"]],
            "hypotheses": [], "next_run_source_id": "", "gaps": []}
    draft = _draft(engine, args)
    assert not draft["verification"]["comparison_coverage"]["complete"]
    assert "Incomplete saved shadow comparison" in draft["answer"]
    assert draft["grounded_publication"]["recommendation"] is None
    reviewed = review_grounded(engine, {"draft": draft, "question": "All saved controls", "round": 1})["report"]
    assert not finalize_grounded(engine, reviewed)["evidence_sufficient"]


def test_single_saved_trial_quarantines_optional_shapes_and_bounds_known_selection(source):
    cards = build_grounded_facts(source)
    args = {**selection(source), "fact_ids": [card["id"] for card in cards],
            "gaps": "Malformed optional gaps", "hypotheses": {"text": "Wrong shape"}}
    draft = _draft(engine_for(source), args)
    assert len(args["fact_ids"]) > 12 and len(draft["claims"]) == 12
    assert len(draft["model_interpretation"]["rejected_annotations"]) == 2
    assert draft["model_interpretation"]["hypotheses"] == draft["model_interpretation"]["gaps"] == []
    assert draft["fact_selection"]["omitted_optional_fact_ids"]
    with pytest.raises(ValueError, match="Unknown grounded fact"):
        _draft(engine_for(source), {**args, "fact_ids": args["fact_ids"] + ["Funknown"]})


@pytest.mark.parametrize("status,execution", [("failed", "failed"), ("partial", "timed_out")])
def test_single_saved_incomplete_trial_has_no_positive_measurements_or_executable_proposal(source, status, execution):
    from fusion_agent.research.shadow_analysis import attach_execution_status

    document = attach_execution_status(json.loads(source["content"]), execution)
    assert document["status"] == status
    encode(source, document)
    args = {"fact_ids": [], "excerpt_ids": [build_excerpt_catalogue([web_source()])[0]["id"]],
            "hypotheses": [], "next_run_source_id": "", "gaps": []}
    draft = _draft(engine_for(source), args)
    assert [card["fact_type"] for card in draft["claims"]] == ["trial_status"]
    assert all(card["provenance"]["origin"] == "saved_shadow_snapshot" for card in draft["claims"])
    proposal = draft["grounded_publication"]["recommendation"]
    if status == "failed":
        assert proposal["code"] == "diagnose_before_comparison" and proposal["request"] is None
    else:
        assert proposal is None and not draft["verification"]["comparison_coverage"]["complete"]


def test_multiple_saved_trials_never_manufacture_a_pooled_mandatory_bundle(source):
    other = copy.deepcopy(source)
    other["id"] = "E002"
    document = json.loads(other["content"])
    document["provenance"]["trial_id"] = "other-trial"
    encode(other, document)
    engine = engine_for(source)
    engine.sources[other["id"]] = other
    engine.read_ids.add(other["id"])
    draft = _draft(engine, selection(source))
    assert len(draft["claims"]) == 1
    assert draft["verification"]["comparison_coverage"]["source_ids"] == []
    assert draft["fact_selection"]["automatically_included_fact_ids"] == []
    assert draft["grounded_publication"]["recommendation"] is None


def test_single_saved_multiseed_study_never_silently_selects_one_seed(source):
    document = json.loads(source["content"])
    document["expected_training_seeds"].append(43)
    encode(source, document)
    args = {"fact_ids": [], "excerpt_ids": [], "hypotheses": [], "next_run_source_id": "", "gaps": []}
    draft = _draft(engine_for(source), args)
    coverage = draft["verification"]["comparison_coverage"]
    assert coverage["scope"] == "multiple_seed_saved_selection" and coverage["source_ids"] == []
    assert coverage["required_fact_ids"] == [] and not draft["claims"]


def test_annotation_quarantine_and_overflow_are_visible_in_writer_trace(source):
    from fusion_agent.research.grounded_writer import write_grounded

    current, events = current_source(source), []

    class MalformedAnnotations:
        def complete(self, messages, tools, tool_choice="auto"):
            payload = json.loads(messages[1]["content"])
            assert tools[0]["function"]["parameters"]["properties"]["fact_ids"]["maxItems"] == 64
            return call("submit_evidence_report", {
                **selection(source), "fact_ids": [card["id"] for card in payload["fact_catalogue"]],
                "gaps": "Wrong shape", "hypotheses": {"text": "Wrong shape"},
            }), {}

    engine = ResearchEngine([source, current, web_source()], MalformedAnnotations(), None, events.append)
    engine.read_ids.update((source["id"], current["id"]))
    draft = write_grounded(engine, {"question": "All four current controls"})["draft"]
    assert len(draft["claims"]) == 12
    assert len([event for event in events if event["kind"] == "optional_annotation_rejected"]) == 2
    assert not any(event["kind"] == "grounded_selection_rejected" for event in events)
    event = next(event for event in events if event["kind"] == "grounded_draft")
    assert event["fact_selection"]["omitted_optional_fact_ids"]
    assert draft["model_interpretation"]["hypotheses"] == []


def test_failure_before_any_result_is_reportable_and_never_relabelled_success(source):
    engine = engine_for(source)
    current = current_source(source, failed=True)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = selection(source)
    args["fact_ids"] = [next(c["id"] for c in build_grounded_facts(current) if c["fact_type"] == "trial_status")]
    args["next_run_source_id"] = current["id"]
    result = _draft(engine, args)
    assert "failed" in result["answer"]
    assert "0.1" not in result["answer"]
    assert result["grounded_publication"]["recommendation"]["request"] is None
    assert result["grounded_publication"]["recommendation"]["code"] == "diagnose_before_comparison"


def test_current_failure_excludes_saved_positive_facts_and_always_exposes_diagnosis(source):
    current = current_source(source, failed=True)
    engine = engine_for(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = selection(source)
    result = _draft(engine, args)
    assert [card["fact_type"] for card in result["claims"]] == ["trial_status"]
    assert result["fact_selection"]["omitted_optional_fact_ids"] == args["fact_ids"]
    assert "Saved shadow" not in result["answer"]
    assert result["grounded_publication"]["recommendation"]["code"] == "diagnose_before_comparison"
    assert result["grounded_publication"]["recommendation"]["request"] is None


def test_review_receives_exact_literature_context_and_preserves_failed_revision_diagnostic(source):
    class ContextJudge(ApprovingJudge):
        def complete(self, messages, tools, tool_choice="auto"):
            payload = json.loads(messages[1]["content"])
            packet = payload["captured_literature_context"]["claims"][0]["evidence"][0]
            assert "In centralized training" in packet["excerpts"][0]["text"]
            return super().complete(messages, tools, tool_choice)
    engine = engine_for(source, ContextJudge())
    args = selection(source)
    reviewed = review_grounded(engine, {"draft": _draft(engine, args), "question": "What is supported?", "round": 1})["report"]
    reviewed["orchestration"]["reflection_failed"] = True
    result = finalize_grounded(engine, reviewed)
    assert any("revision failed" in line for line in result["limitations"])
    assert result["grounded_publication"]["limitations"] == result["limitations"]


class GroundedTeam(TeamModel):
    def __init__(self):
        super().__init__()
        self.selections = 0
        self.reviews = 0

    def complete(self, messages, tools, tool_choice="auto"):
        names = {t["function"]["name"] for t in tools}
        if names == {"submit_evidence_report"}:
            self.selections += 1
            payload = json.loads(messages[1]["content"])
            cards = payload["fact_catalogue"]
            args = {"fact_ids": [next(c["id"] for c in cards if c["kind"] == "paired_raw_endpoint")],
                    "excerpt_ids": [payload["excerpt_catalogue"][0]["id"]],
                    "hypotheses": [], "next_run_source_id": "", "gaps": []}
            return call("submit_evidence_report", args), {"total_tokens": 10}
        if names == {"review_evidence_selection"}:
            self.reviews += 1
            return call("review_evidence_selection", {"decision": "accept", "gaps": [],
                        "rejected_hypothesis_indices": []}), {"total_tokens": 10}
        return super().complete(messages, tools, tool_choice)


class PrimaryRetrieval(FakeRetrieval):
    def search(self, query):
        result = super().search(query)
        result["results"][0]["url"] = "https://arxiv.org/abs/1802.09477"
        return result


def test_real_langgraph_uses_default_grounded_writer_and_read_only_judge(source):
    source.update(title="Synthetic shadow fixture", uri="results/summary.json")
    model, events = GroundedTeam(), []
    engine = ResearchEngine([source], model, PrimaryRetrieval(), events.append)
    report = engine.run("Inspect the recorded paired cost and a primary source.")
    assert engine.grounded_reporting is True
    assert model.selections == model.reviews == 1
    assert report["grounded_publication"]["status"] == "grounded"
    assert report["answer"] == report["grounded_publication"]["answer"]
    assert "0.12" not in report["answer"]  # Fabricated specialist summary is not factual authority.
    assert report["orchestration"]["framework"] == "LangGraph"
    assert report["judge"]["authority"] == "relevance_and_hypothesis_critique_only"
    assert any(event["kind"] == "grounded_draft" for event in events)
    assert any(event["kind"] == "grounded_review" for event in events)


def test_secondary_summary_cannot_satisfy_primary_slot_even_with_approving_judge(source):
    engine = engine_for(source, ApprovingJudge())
    engine.sources["W002"]["uri"] = "https://tinycto.tv/ml-atlas/literature/fujimoto-2018"
    args = selection(source)
    with pytest.raises(ValueError, match="approved primary-publication"):
        _draft(engine, args)
    args["excerpt_ids"] = []
    report = review_grounded(engine, {"draft": _draft(engine, args), "question": "Require a primary paper.", "round": 1})["report"]
    result = finalize_grounded(engine, report)
    assert result["judge"]["decision"] == "accept"
    assert result["evidence_sufficient"] is False
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["grounded_publication"]["status"] == "insufficient_evidence"
    assert "No eligible extracted primary-publication" in result["stop_reason"]
    assert any("secondary summaries cannot" in line for line in result["limitations"])


def test_writer_offers_only_eligible_references_and_no_empty_enum(source):
    from fusion_agent.research.grounded_writer import write_grounded

    class InspectWriter:
        def __init__(self):
            self.expect_primary = False

        def complete(self, messages, tools, tool_choice="auto"):
            payload = json.loads(messages[1]["content"])
            schema = tools[0]["function"]["parameters"]["properties"]
            assert schema["fact_ids"]["maxItems"] == 64
            literature = schema["excerpt_ids"]
            enum = literature["items"].get("enum")
            args = selection(source)
            if self.expect_primary:
                assert enum == [item["id"] for item in payload["excerpt_catalogue"]]
                assert len(payload["web_sources"]) == 1
                assert all("request" not in item for item in payload["excerpt_catalogue"])
                args["excerpt_ids"] = [enum[0]]
            else:
                assert enum is None and literature["maxItems"] == 0
                assert payload["web_sources"] == []
                assert payload["primary_reference_gap"]
                args["excerpt_ids"] = []
            return call("submit_evidence_report", args), {}

    model = InspectWriter()
    engine = engine_for(source, model)
    engine.sources["W002"]["uri"] = "https://example.org/summary"
    report = write_grounded(engine, {"question": "Primary evidence"})["draft"]
    assert report["grounded_publication"]["status"] == "insufficient_evidence"
    model.expect_primary = True
    engine.sources["W002"]["uri"] = "https://arxiv.org/abs/1802.09477"
    report = write_grounded(engine, {"question": "Primary evidence"})["draft"]
    assert report["grounded_publication"]["status"] == "grounded"


def test_final_literature_turn_can_extract_discovered_primary_without_extra_model_turn(source):
    class LateReader(GroundedTeam):
        def complete(self, messages, tools, tool_choice="auto"):
            if messages[0]["content"].startswith("Agent role: literature_researcher\n"):
                self.counts["literature_researcher"] += 1
                step = self.counts["literature_researcher"]
                if step == 1:
                    self.parallel_barrier.wait(timeout=4)
                if step < 6:
                    return call("search_literature", {"query": "TD3 primary methods paper", "reason": "Find the paper."}), {}
                assert step == 6
                assert {tool["function"]["name"] for tool in tools} == {"read_page", "submit_findings"}
                final = json.loads(messages[-1]["content"])
                assert final["eligible_primary_source_ids"]
                return call("read_page", {"source_id": final["eligible_primary_source_ids"][0], "reason": "Capture the discovered primary paper."}), {}
            return super().complete(messages, tools, tool_choice)

    source.update(title="Synthetic shadow fixture", uri="results/summary.json")
    model, retrieval, events = LateReader(), PrimaryRetrieval(), []
    report = ResearchEngine([source], model, retrieval, events.append).run("Inspect saved costs and a primary reference.")
    assert model.counts["literature_researcher"] == 6
    assert len([entry for entry in retrieval.calls if entry[0] == "extract"]) == 1
    assert report["grounded_publication"]["literature_excerpts"]
    literature_complete = [event for event in events if event["kind"] == "agent_complete" and event.get("agent") == "literature_researcher"]
    assert "unresolved gaps" in literature_complete[-1]["message"]
    assert literature_complete[-1]["evidence_ids"]
