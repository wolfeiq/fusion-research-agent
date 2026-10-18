"""A judged citation defect gets one bounded repair using captured evidence."""

import json
from copy import deepcopy

import pytest
from test_research_graph import report, setup_engine
from test_shadow_graph import setup as setup_shadow

from fusion_agent.research.graph import _preserve_protected_claims, _protected_claims


def install_reflection_model(monkeypatch, engine, model, *, repair=True, fail_draft=False):
    original = model.complete
    reflection_calls = []

    def complete(messages, tools, tool_choice="auto"):
        role = messages[0]["content"].split("\n", 1)[0].removeprefix("Agent role: ")
        payload = json.loads(messages[1]["content"])
        if role == "writer" and payload.get("phase") == "writer_reflection":
            reflection_calls.append(messages)
            assert payload["reviewed_draft"]["claims"][-1]["id"] == "C_BAD"
            assert payload["reviewed_draft"]["claims"][-1]["verification"]["status"] == "unsupported"
            assert payload["judge_gaps"]
            assert payload["reviewed_draft"]["judge"]["decision"] != "accept"
            allowed = {tool["function"]["name"] for tool in tools}
            assert not allowed & {"search_literature", "read_experiment", "test_shadow"}
            if fail_draft:
                return {"role": "assistant", "content": "No valid draft supplied."}, {}
            if len(reflection_calls) == 1:
                calls = [
                    ("cite_experiment_value", {"source_id": "E001", "path": ["cost"], "reason": "Repair the exact recorded-cost citation."}),
                    ("assess_evidence", {"enough": True, "gaps": [], "reason": "The missing pointer can be repaired from captured evidence."}),
                ]
            else:
                assert "finish_report" in allowed
                fixed = report(model.web_id)
                citation = next(json.loads(message["content"]) for message in messages
                                if message["role"] == "tool" and '"locator": "/cost"' in message["content"])
                fixed["claims"].append({"id": "C_BAD", "text": "The recorded cost is 0.12.",
                                        "kind": "observation", "confidence": "high", "evidence": [citation]})
                if not repair:
                    fixed["claims"][-1]["evidence"][0]["locator"] = "/still_missing"
                calls = [("finish_report", fixed)]
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"reflection_{len(reflection_calls)}_{index}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}
                for index, (name, args) in enumerate(calls)
            ]}, {"total_tokens": 1}

        response, usage = original(messages, tools, tool_choice)
        call = response["tool_calls"][0]["function"]
        args = json.loads(call["arguments"])
        if role == "writer" and call["name"] == "finish_report":
            initial = json.loads(args["report_json"])
            initial["claims"].append({"id": "C_BAD", "text": "The recorded cost is 0.12.",
                                      "kind": "observation", "confidence": "high", "evidence": [{
                                          "source_id": "E001", "quote": "0.12", "locator": "/missing",
                                      }]})
            args = {"report_json": json.dumps(initial)}
        elif call["name"] == "review_claims":
            repaired = bool(reflection_calls) and repair
            # Review only submitted claim IDs; a repaired draft may omit a prior
            # shadow finding while retaining its source for the proposal.
            ids = {claim["id"] for claim in payload["claims"]}
            args["verdicts"] = [verdict for verdict in args["verdicts"] if verdict["id"] in ids]
            args["verdicts"].append({"id": "C_BAD", "verdict": "supported" if repaired else "unsupported",
                                     "reason": "The pointer covers this claim." if repaired else "The pointer does not resolve.",
                                     "all_clauses_supported": repaired, "scope_matches": repaired,
                                     "unsupported_clauses": [] if repaired else ["The cost citation is unresolved."]})
            args["decision"] = "accept" if repaired else "insufficient"
            args["gaps"] = [] if repaired else ["Resolve the missing cost pointer from the captured record."]
        call["arguments"] = json.dumps(args)
        return response, usage

    monkeypatch.setattr(model, "complete", complete)
    return reflection_calls


@pytest.mark.parametrize("repair", [True, False])
def test_citation_reflection_rechecks_once_and_never_overrides_a_persistent_failure(monkeypatch, repair):
    engine, model, _, events = setup_engine()
    reflections = install_reflection_model(monkeypatch, engine, model, repair=repair)
    result = engine.run("Repair any citation gaps using captured evidence, without more research.")
    assert len(reflections) == 2
    assert model.counts["planner"] == 1 and model.counts["judge"] == 2
    assert engine.searches == engine.extracts == 1
    assert result["orchestration"]["writer_reflections"] == 1
    assert result["orchestration"]["rounds"] == 1
    assert result["orchestration"]["status"] == ("accepted" if repair else "insufficient_evidence")
    assert result["evidence_sufficient"] is repair
    assert any(event["kind"] == "writer_reflection" for event in events)
    assert engine.max_turns == 28 and engine.max_calls == 48


def test_shadow_result_is_reused_for_reflection_without_another_physics_execution(monkeypatch):
    engine, model, events, executions = setup_shadow()
    reflections = install_reflection_model(monkeypatch, engine, model)
    result = engine.run("Retain the measured shadow while repairing the final citation.")
    assert len(reflections) == 2
    assert len(executions) == len(model.decision_payloads) == 1
    assert result["orchestration"]["status"] == "accepted"
    assert model.shadow_id in {source["id"] for source in json.loads(reflections[0][1]["content"])["sources"]}
    assert sum(event["kind"] == "writer_reflection" for event in events) == 1


def test_exhausted_reflection_retains_already_judged_insufficient_report(monkeypatch):
    engine, model, _, events = setup_engine()
    reflections = install_reflection_model(monkeypatch, engine, model, fail_draft=True)
    result = engine.run("Retain judged evidence if the bounded repair cannot finish.")
    assert len(reflections) == 3
    assert model.counts["judge"] == 1
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["claims"][-1]["review"]["verdict"] == "unsupported"
    assert any(event["kind"] == "writer_reflection_failed" for event in events)


def test_reflection_requires_four_remaining_calls_including_independent_judge(monkeypatch):
    engine, model, _, events = setup_engine()
    reflections = install_reflection_model(monkeypatch, engine, model)
    original = model.complete

    def near_budget_review(messages, tools, tool_choice="auto"):
        response = original(messages, tools, tool_choice)
        if response[0].get("tool_calls", [{}])[0].get("function", {}).get("name") == "review_claims":
            # Inject the boundary at the finished review; preserve the ordinary
            # collection phase so its independent reserve is not under test here.
            engine.max_turns = engine.model_calls + 3
        return response

    monkeypatch.setattr(model, "complete", near_budget_review)
    result = engine.run("Preserve the judged report when a repair cannot fit its budget.")
    assert not reflections
    assert model.counts["judge"] == 1
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["orchestration"]["writer_reflections"] == 0
    deferred = next(event for event in events if event["kind"] == "writer_reflection_deferred")
    assert deferred["remaining_model_calls"] == 3
    assert deferred["minimum_reflection_calls"] == 4


@pytest.mark.parametrize("omit", [False, True])
def test_reflection_cannot_damage_previously_supported_citation_and_all_claims_are_rejudged(monkeypatch, omit):
    engine, model, _, events = setup_engine()
    reflections = install_reflection_model(monkeypatch, engine, model)
    original = model.complete
    source_snapshot = None

    def rewrite_supported_claim(messages, tools, tool_choice="auto"):
        nonlocal source_snapshot
        response, usage = original(messages, tools, tool_choice)
        payload = json.loads(messages[1]["content"])
        call = response["tool_calls"][0]["function"]
        if payload.get("phase") == "writer_reflection":
            if source_snapshot is None:
                source_snapshot = deepcopy(engine.sources)
            assert {claim["id"] for claim in payload["protected_claims"]} == {"C1", "C2"}
            for claim in payload["protected_claims"]:
                assert set(claim) == {"id", "text", "kind", "confidence", "evidence"}
                assert all(set(citation) == {"source_id", "quote", "locator"} for citation in claim["evidence"])
            if call["name"] == "finish_report":
                draft = json.loads(call["arguments"])
                if omit:
                    draft["claims"] = [claim for claim in draft["claims"] if claim["id"] != "C2"]
                else:
                    claim = next(claim for claim in draft["claims"] if claim["id"] == "C2")
                    claim["text"] = "An invented replacement conclusion."
                    claim["evidence"][0].update(quote="Invented quote", locator="chars:395:480")
                call["arguments"] = json.dumps(draft)
        return response, usage

    monkeypatch.setattr(model, "complete", rewrite_supported_claim)
    result = engine.run("Repair the flagged claim while keeping the checked passage immutable.")
    assert len(reflections) == 2 and model.counts["judge"] == 2
    assert result["orchestration"]["status"] == "accepted"
    saved = report(model.web_id)["claims"][1]
    actual = next(claim for claim in result["claims"] if claim["id"] == "C2")
    assert actual["text"] == saved["text"]
    assert {key: actual["evidence"][0][key] for key in saved["evidence"][0]} == saved["evidence"][0]
    preserved = next(event for event in events if event["kind"] == "writer_protected_claims_preserved")
    assert preserved["fresh_review_required"] is True
    assert "C2" in preserved["restored_omitted_claim_ids" if omit else "replaced_submitted_claim_ids"]
    assert {claim["id"] for claim in json.loads(model.histories["judge"][-1][1]["content"])["claims"]} == {"C1", "C2", "C_BAD"}
    assert engine.sources == source_snapshot


def test_fresh_judge_can_reject_a_protected_claim(monkeypatch):
    engine, model, _, events = setup_engine()
    reflections = install_reflection_model(monkeypatch, engine, model)
    original = model.complete

    def reject_on_recheck(messages, tools, tool_choice="auto"):
        response, usage = original(messages, tools, tool_choice)
        call = response["tool_calls"][0]["function"]
        if call["name"] == "review_claims" and reflections:
            args = json.loads(call["arguments"])
            claim = next(item for item in args["verdicts"] if item["id"] == "C2")
            claim.update(verdict="unsupported", all_clauses_supported=False, scope_matches=False,
                         unsupported_clauses=["The passage has narrower scope."], reason="The passage has narrower scope.")
            args.update(decision="insufficient", gaps=["The previously accepted paper claim needs a narrower scope."])
            call["arguments"] = json.dumps(args)
        return response, usage

    monkeypatch.setattr(model, "complete", reject_on_recheck)
    result = engine.run("Recheck every claim independently, including previously supported claims.")
    assert any(event["kind"] == "writer_protected_claims_preserved" for event in events)
    assert result["orchestration"]["status"] == "insufficient_evidence"
    rejected = next(claim for claim in result["claims"] if claim["id"] == "C2")
    assert rejected["review"]["verdict"] == "unsupported"
    assert rejected["text"] not in result["answer"]


def test_reflection_does_not_protect_unknown_citations_in_a_flagged_claim(monkeypatch):
    engine, model, _, events = setup_engine()
    install_reflection_model(monkeypatch, engine, model)
    original = model.complete

    def unknown_source(messages, tools, tool_choice="auto"):
        response, usage = original(messages, tools, tool_choice)
        call = response["tool_calls"][0]["function"]
        if json.loads(messages[1]["content"]).get("phase") == "writer_reflection" and call["name"] == "finish_report":
            draft = json.loads(call["arguments"])
            draft["claims"][-1]["evidence"][0]["source_id"] = "UNKNOWN_SOURCE"
            call["arguments"] = json.dumps(draft)
        return response, usage

    monkeypatch.setattr(model, "complete", unknown_source)
    result = engine.run("Keep unknown sources unsupported even if the model calls them supported.")
    assert result["orchestration"]["status"] == "insufficient_evidence"
    bad = next(claim for claim in result["claims"] if claim["id"] == "C_BAD")
    assert bad["verification"]["status"] == "unsupported"
    assert all("C_BAD" not in event["protected_claim_ids"] for event in events if event["kind"] == "writer_protected_claims_preserved")


def test_protected_merge_rejects_duplicate_ids_and_does_not_invent_protected_claims():
    draft = report()
    reviewed = deepcopy(draft)
    for claim in reviewed["claims"]:
        claim.update(verification={"status": "unsupported"}, review={"verdict": "unsupported"})
    assert _protected_claims(reviewed) == {}
    merged, audit = _preserve_protected_claims(draft, {})
    assert merged == draft and audit["protected_claim_ids"] == []
    reviewed["claims"][0].update(verification={"status": "traceable"}, review={"verdict": "supported"})
    protected = _protected_claims(reviewed)
    duplicate = deepcopy(draft)
    duplicate["claims"].append(deepcopy(duplicate["claims"][0]))
    with pytest.raises(ValueError, match="Claim IDs must be unique"):
        _preserve_protected_claims(duplicate, protected)
    oversized = deepcopy(draft)
    oversized["claims"][0]["text"] = "x" * 50_000
    with pytest.raises(ValueError, match="50000-character"):
        _preserve_protected_claims(oversized, protected)


@pytest.mark.parametrize("envelope", ["flat", "object", "encoded", "legacy"])
def test_reflection_accepts_existing_report_transports_and_keeps_normalization_audit(monkeypatch, envelope):
    engine, model, _, events = setup_engine()
    install_reflection_model(monkeypatch, engine, model)
    original = model.complete

    def wrap_report(messages, tools, tool_choice="auto"):
        response, usage = original(messages, tools, tool_choice)
        call = response["tool_calls"][0]["function"]
        if json.loads(messages[1]["content"]).get("phase") == "writer_reflection" and call["name"] == "finish_report":
            draft = json.loads(call["arguments"])
            if envelope == "object":
                args = {"report": draft}
            elif envelope == "encoded":
                args = {"report": json.dumps(draft)}
            elif envelope == "legacy":
                args = {"report_json": json.dumps(draft)}
            else:
                args = draft
            call["arguments"] = json.dumps(args)
        return response, usage

    monkeypatch.setattr(model, "complete", wrap_report)
    result = engine.run("Accept existing transports while preserving checked claims and rechecking the whole report.")
    assert result["orchestration"]["status"] == "accepted"
    assert model.counts["judge"] == 2
    normalized = [event for event in events if event.get("transport") == "json_string_in_report"]
    assert len(normalized) == int(envelope == "encoded")
    if normalized:
        assert normalized[0]["decoded_layers"] == 1


@pytest.mark.parametrize("args", [
    {"report": report(), "extra": True}, {"report": report(), "report_json": json.dumps(report())},
    {"report": json.dumps(report()) + " trailing"}, {"report": json.dumps(json.dumps(report()))},
    {"report": {**report(), "unknown": "field"}},
    {"report": {**report(), "claims": json.dumps(report()["claims"])}},
    {"report": '{"answer": NaN}'}, {"report_json": json.dumps(report()) + " {}"},
    {"report": json.dumps({"answer": "x" * 50_001})},
])
def test_reflection_transport_never_accepts_unknown_fields_nested_arrays_or_malformed_json(args):
    with pytest.raises(ValueError):
        _preserve_protected_claims(args, {})
