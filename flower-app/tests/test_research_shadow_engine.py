"""The model receives one typed shadow capability, never arbitrary execution."""
import hashlib
import json

import pytest

from fusion_agent.research.engine import TOOLS, ResearchEngine


def engine(executor=None, events=None):
    content = '{"status":"complete"}'
    source = {"id": "E001", "title": "Result", "kind": "experiment", "content": content,
              "sha256": hashlib.sha256(content.encode()).hexdigest(), "uri": "saved:result", "metadata": {}}
    result = ResearchEngine([source], None, None, events.append if events is not None else lambda event: None,
                            shadow_executor=executor)
    result.execute("plan_research", {"questions": ["Does slower actor learning change endpoint cost?"], "reason": "Test a measured concern."})
    return result


ARGS = {"source_id": "E001", "change": "td3_actor_lr", "value": .00005, "seed": 17,
        "reason": "Compare actor drift with paired controls."}


def test_shadow_is_absent_unless_enabled_and_role_scoped():
    model = engine()
    with model.agent("federation_strategist"), pytest.raises(ValueError, match="not enabled"):
        model.execute("test_shadow", ARGS)
    model = engine(lambda *_: pytest.fail("Forbidden role executed shadow"))
    with model.agent("td3_specialist"), pytest.raises(ValueError, match="Only the federation strategist"):
        model.execute("test_shadow", ARGS)


def test_reads_evidence_before_one_attempt_and_registers_measured_source():
    calls = []

    def execute(change, value, *, seed):
        calls.append((change, value, seed))
        return {"status": "completed", "comparison": {"recommendation": {"efficacy_claim": False}}, "policy_activated": False}

    model = engine(execute)
    with model.agent("federation_strategist"):
        with pytest.raises(ValueError, match="Read the selected experiment"):
            model.execute("test_shadow", ARGS)
        model.execute("read_experiment", {"source_id": "E001", "reason": "Inspect the prior evidence"})
        result = model.execute("test_shadow", ARGS)
        with pytest.raises(ValueError, match="already been used"):
            model.execute("test_shadow", ARGS)
    assert calls == [("td3_actor_lr", .00005, 17)]
    assert result["id"] in model.read_ids
    assert result["metadata"]["adapter"] == "shadow_execution"
    assert hashlib.sha256(result["content"].encode()).hexdigest() == result["sha256"]
    assert json.loads(result["content"])["policy_activated"] is False


def test_failed_attempt_consumes_budget_without_retrying():
    def fail(*_, **__):
        raise ValueError("An independent shadow worker is already active")

    model = engine(fail)
    with model.agent("federation_strategist"):
        model.execute("read_experiment", {"source_id": "E001", "reason": "Read before choosing"})
        with pytest.raises(ValueError, match="already active"):
            model.execute("test_shadow", ARGS)
        with pytest.raises(ValueError, match="already been used"):
            model.execute("test_shadow", ARGS)


@pytest.mark.parametrize("change,value", [("shell", "cmd"), ("td3_actor_lr", 1), ("sac_mix", float("nan"))])
def test_invalid_model_configuration_never_reaches_worker(change, value):
    model = engine(lambda *_: pytest.fail("Invalid candidate reached worker"))
    with model.agent("federation_strategist"), pytest.raises(ValueError):
        model.execute("test_shadow", {**ARGS, "change": change, "value": value})


def test_new_tool_calls_require_one_bounded_seed_without_expanding_rounds():
    schema = next(tool["function"]["parameters"] for tool in TOOLS
                  if tool["function"]["name"] == "test_shadow")
    assert "seed" in schema["required"]
    assert schema["properties"]["seed"] == {"type": "integer", "minimum": 0, "maximum": 1_000_000}
    assert "seeds" not in schema["properties"] and "rounds" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_missing_seed_is_rejected_before_attempt_without_silent_default():
    calls, events = [], []

    def execute(change, value, *, seed):
        calls.append(seed)
        return {"status": "completed", "policy_activated": False}

    model = engine(execute, events)
    with model.agent("federation_strategist"):
        model.execute("read_experiment", {"source_id": "E001", "reason": "Inspect before trial"})
        with pytest.raises(ValueError, match="explicit development seed"):
            model.execute("test_shadow", {key: value for key, value in ARGS.items() if key != "seed"})
        assert not model.shadow_attempted and not calls
        assert not any(event["kind"] == "shadow_start" for event in events)
        model.execute("test_shadow", {**ARGS, "seed": 29})
    assert calls == [29]


@pytest.mark.parametrize("seed", [True, False, "29", 29.0, None, -1, 1_000_001, [29]])
def test_invalid_seed_does_not_consume_attempt_or_reach_executor(seed):
    calls, events = [], []

    def execute(change, value, *, seed):
        calls.append((change, value, seed))
        return {"status": "completed", "policy_activated": False}

    model = engine(execute, events)
    with model.agent("federation_strategist"):
        model.execute("read_experiment", {"source_id": "E001", "reason": "Inspect before trial"})
        with pytest.raises(ValueError, match="development seeds"):
            model.execute("test_shadow", {**ARGS, "seed": seed})
        assert not model.shadow_attempted and not calls
        assert not any(event["kind"] == "shadow_start" for event in events)
        model.execute("test_shadow", {**ARGS, "seed": 29})
        with pytest.raises(ValueError, match="already been used"):
            model.execute("test_shadow", {**ARGS, "seed": 43})
    assert calls == [("td3_actor_lr", .00005, 29)]
    event = next(event for event in events if event["kind"] == "shadow_start")
    assert event["seeds"] == [29] and event["rounds"] == 1


@pytest.mark.parametrize("seed", [0, 1_000_000])
def test_explicit_seed_boundary_is_preserved(seed):
    calls = []

    def execute(change, value, *, seed):
        calls.append(seed)
        return {"status": "completed", "policy_activated": False}

    model = engine(execute)
    with model.agent("federation_strategist"):
        model.execute("read_experiment", {"source_id": "E001", "reason": "Inspect before trial"})
        model.execute("test_shadow", {**ARGS, "seed": seed})
    assert calls == [seed]
