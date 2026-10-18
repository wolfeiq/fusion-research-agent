"""Larger pure citation batches preserve validation and the global tool budget."""

import hashlib
import json

import pytest
from test_research_graph import report, setup_engine

from fusion_agent.research.graph import _batch_limit, _calls


def citation_calls(model, count):
    return [{"id": f"lookup_{index}", "type": "function", "function": {
        "name": "cite_experiment_value" if index < count - 2 else "cite_web_excerpt",
        "arguments": json.dumps({"source_id": "E001", "path": ["measurements", index],
                                  "reason": "Read a recorded endpoint."} if index < count - 2 else {
            "source_id": model.web_id, "anchor": "Clipping can limit" if index % 2 else "large updates.",
            "max_words": 5, "reason": "Read the exact captured passage.",
        }),
    }} for index in range(count)]


def install_batch(monkeypatch, engine, model, calls_factory, *, expect_rejection=False, cap=None):
    source = engine.sources["E001"]
    content = json.loads(source["content"])
    content["measurements"] = [0.1 + index / 1000 for index in range(24)]
    source["content"] = json.dumps(content)
    source["sha256"] = hashlib.sha256(source["content"].encode()).hexdigest()
    original_complete, original_execute = model.complete, engine.execute
    state = {"injected": False, "checked": False, "executed": [], "payloads": []}

    def complete(messages, tools, tool_choice="auto"):
        if messages[0]["content"].startswith("Agent role: writer\n"):
            payload = json.loads(messages[1]["content"])
            state["payloads"].append(payload)
            if not state["injected"]:
                state["injected"] = True
                if cap is not None:
                    engine.max_calls = engine.calls + cap
                return {"role": "assistant", "content": "", "tool_calls": calls_factory(model)}, {"total_tokens": 1}
            if not state["checked"]:
                state["checked"] = True
                if expect_rejection:
                    feedback = json.loads(messages[-1]["content"])
                    assert messages[-1]["role"] == "user"
                    assert "entire previous tool batch was rejected" in feedback["instruction"]
                    assert "lookup_" not in json.dumps(messages)
                    assert not state["executed"]
                    state["feedback"] = feedback
                else:
                    replies = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
                    assert len(replies) == 14
                    assert all(set(reply) == {"source_id", "locator", "quote"} for reply in replies)
                    assert replies[0]["locator"] == "/measurements/0"
                    assert replies[-1]["locator"].startswith("chars:")
        return original_complete(messages, tools, tool_choice)

    def execute(name, args):
        if args.get("reason") in {"Read a recorded endpoint.", "Read the exact captured passage."}:
            state["executed"].append(name)
        return original_execute(name, args)

    monkeypatch.setattr(model, "complete", complete)
    monkeypatch.setattr(engine, "execute", execute)
    return state


def test_writer_executes_fourteen_local_and_web_citations_and_still_finishes(monkeypatch):
    engine, model, _, events = setup_engine()
    state = install_batch(monkeypatch, engine, model, lambda model: citation_calls(model, 14))
    result = engine.run("Cite the measured values and captured literature.")
    assert len(state["executed"]) == 14
    assert result["orchestration"]["status"] == "accepted"
    assert model.counts["judge"] == 1
    assert not any(event["kind"] == "agent_batch_rejected" for event in events)
    budgets = [payload["tool_budget"] for payload in state["payloads"]]
    assert budgets[1]["remaining_calls"] == budgets[0]["remaining_calls"] - 14
    assert budgets[1]["reserved_terminal_calls"] == 2
    assert budgets[2]["reserved_terminal_calls"] == 1
    assert engine.calls <= engine.max_calls == 48
    assert engine.max_turns == 28


@pytest.mark.parametrize("invalid", ["too_many", "malformed", "finish_report", "read_experiment", "test_shadow"])
def test_invalid_or_mixed_large_writer_batch_executes_nothing_and_recovers(monkeypatch, invalid):
    engine, model, _, events = setup_engine()

    def batch(model):
        calls = citation_calls(model, 25 if invalid == "too_many" else 14 if invalid == "malformed" else 9)
        if invalid == "malformed":
            calls[-1]["function"]["arguments"] = {}
        elif invalid not in {"too_many", "malformed"}:
            calls[-1]["function"] = {"name": invalid, "arguments": json.dumps(report(model.web_id) if invalid == "finish_report" else {})}
        return calls

    state = install_batch(monkeypatch, engine, model, batch, expect_rejection=True)
    result = engine.run("Reject invalid entire batches without losing the final report.")
    assert result["orchestration"]["status"] == "accepted"
    assert state["checked"] and not state["executed"]
    expected_limit = 24 if invalid in {"too_many", "malformed"} else 8
    assert state["feedback"]["maximum_tool_calls"] == expected_limit
    rejection = next(event for event in events if event["kind"] == "agent_batch_rejected")
    assert rejection["maximum_tool_calls"] == expected_limit


def test_citation_capacity_preserves_global_calls_needed_to_assess_and_finish(monkeypatch):
    engine, model, _, events = setup_engine()
    state = install_batch(monkeypatch, engine, model, lambda model: citation_calls(model, 14),
                          expect_rejection=True, cap=15)
    result = engine.run("Leave enough tool budget for assessment and submission.")
    assert result["orchestration"]["status"] == "accepted"
    assert not state["executed"]
    assert state["feedback"]["remaining_tool_calls"] == 15
    assert state["feedback"]["reserved_terminal_tool_calls"] == 2
    assert state["feedback"]["maximum_affordable_citation_calls"] == 13
    assert state["payloads"][1]["tool_budget"]["maximum_citation_calls_this_turn"] == 13
    assert engine.calls <= engine.max_calls
    assert any("available global tool budget" in event.get("validation_error", "") for event in events)


@pytest.mark.parametrize("role", ["planner", "td3_specialist", "sac_specialist", "federation_strategist", "literature_researcher", "judge"])
def test_larger_citation_capacity_is_never_available_to_other_roles(role):
    _, model, _, _ = setup_engine()
    message = {"tool_calls": citation_calls(model, 14)}
    limit = _batch_limit(message, role, {"cite_experiment_value", "cite_web_excerpt"})
    assert limit == 8
    with pytest.raises(ValueError, match="batch limit"):
        _calls(message, limit=limit)


def test_writer_terminal_tools_cannot_reopen_the_larger_citation_allowance():
    _, model, _, _ = setup_engine()
    message = {"tool_calls": citation_calls(model, 14)}
    assert _batch_limit(message, "writer", {"finish_report"}) == 8
    message["tool_calls"][-1]["function"]["name"] = []
    assert _batch_limit(message, "writer", {"cite_experiment_value", "cite_web_excerpt"}) == 8
