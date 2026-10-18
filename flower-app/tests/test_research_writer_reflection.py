"""A judged citation defect gets one bounded repair using captured evidence."""

import json

import pytest
from test_research_graph import report, setup_engine
from test_shadow_graph import setup as setup_shadow


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
