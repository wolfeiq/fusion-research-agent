"""The model receives one typed shadow capability, never arbitrary execution."""
import hashlib
import json

import pytest

from fusion_agent.research.engine import ResearchEngine


def engine(executor=None):
    content = '{"status":"complete"}'
    source = {"id": "E001", "title": "Result", "kind": "experiment", "content": content,
              "sha256": hashlib.sha256(content.encode()).hexdigest(), "uri": "saved:result", "metadata": {}}
    result = ResearchEngine([source], None, None, lambda event: None, shadow_executor=executor)
    result.execute("plan_research", {"questions": ["Does slower actor learning change endpoint cost?"], "reason": "Test a measured concern."})
    return result


ARGS = {"source_id": "E001", "change": "td3_actor_lr", "value": .00005, "reason": "Compare actor drift with paired controls."}


def test_shadow_is_absent_unless_enabled_and_role_scoped():
    model = engine()
    with model.agent("federation_strategist"), pytest.raises(ValueError, match="not enabled"):
        model.execute("test_shadow", ARGS)
    model = engine(lambda *_: pytest.fail("Forbidden role executed shadow"))
    with model.agent("td3_specialist"), pytest.raises(ValueError, match="Only the federation strategist"):
        model.execute("test_shadow", ARGS)


def test_reads_evidence_before_one_attempt_and_registers_measured_source():
    calls = []

    def execute(change, value):
        calls.append((change, value))
        return {"status": "completed", "comparison": {"recommendation": {"efficacy_claim": False}}, "policy_activated": False}

    model = engine(execute)
    with model.agent("federation_strategist"):
        with pytest.raises(ValueError, match="Read the selected experiment"):
            model.execute("test_shadow", ARGS)
        model.execute("read_experiment", {"source_id": "E001", "reason": "Inspect the prior evidence"})
        result = model.execute("test_shadow", ARGS)
        with pytest.raises(ValueError, match="already been used"):
            model.execute("test_shadow", ARGS)
    assert calls == [("td3_actor_lr", .00005)]
    assert result["id"] in model.read_ids
    assert result["metadata"]["adapter"] == "shadow_execution"
    assert hashlib.sha256(result["content"].encode()).hexdigest() == result["sha256"]
    assert json.loads(result["content"])["policy_activated"] is False


def test_failed_attempt_consumes_budget_without_retrying():
    def fail(*_):
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
