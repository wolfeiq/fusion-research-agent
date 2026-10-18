"""Only limitations prose may cross the explicit string/list transport boundary."""

import json
from copy import deepcopy

import pytest
from test_research_native_report import engine, report

from fusion_agent.research.verification import normalize_report_transport


def proposal():
    return {
        "title": "Repeat the paired control comparison",
        "hypothesis": "Clipping may alter measured controller cost.",
        "changes": "Change only the clipping fraction from 0.1 to 0.001.",
        "controls": ["local", "reset-only", "uniform", "clipped"],
        "metrics": ["raw endpoint cost", "gate fallback endpoint cost"],
        "budget": "112 episodes: 64 training and 48 evaluation across two seeds.",
        "stop_rule": "Stop if the recorded source hashes change.",
        "evidence_ids": ["E001"], "limitations": "Development only.",
    }


def with_proposal():
    draft = report()
    draft["next_runs"] = [proposal()]
    return draft


@pytest.mark.parametrize("envelope", ["flat", "object", "encoded"])
def test_known_prose_shapes_preserve_every_character_order_and_other_fields(envelope):
    draft = with_proposal()
    paragraphs = ["  Seed 17 only.\n", "\tNo efficacy claim.  "]
    draft["limitations"] = "\n  No new confirmation data.\t "
    draft["next_runs"][0]["limitations"] = paragraphs
    original = deepcopy(draft)
    normalized, changes = normalize_report_transport(draft)
    assert draft == original
    expected = deepcopy(draft)
    expected["limitations"] = [draft["limitations"]]
    expected["next_runs"][0]["limitations"] = "\n\n".join(paragraphs)
    assert normalized == expected
    assert changes == [
        {"path": "/limitations", "source_type": "string", "target_type": "array",
         "item_count": 1, "source_item_lengths": [len(draft["limitations"])]},
        {"path": "/next_runs/0/limitations", "source_type": "array", "target_type": "string",
         "item_count": 2, "separator": "\n\n", "source_item_lengths": list(map(len, paragraphs))},
    ]
    args = draft if envelope == "flat" else {"report": draft if envelope == "object" else json.dumps(draft)}
    events = []
    runner = engine()
    runner.emit = events.append
    result = runner.execute("finish_report", args)["report"]
    assert draft == original
    assert result["limitations"][0] == original["limitations"]
    assert result["next_runs"][0]["limitations"] == "\n\n".join(paragraphs)
    assert result["next_runs"][0]["budget"] == original["next_runs"][0]["budget"]
    assert result["claims"][0]["evidence"][0]["quote"] == "0.12"
    assert result["verification"]["transport_normalizations"] == changes
    assert result["verification"]["semantic_review_required"] is True
    event = next(event for event in events if event.get("transport") == "limitations_prose_shapes")
    assert event["normalized_fields"] == changes
    assert "Seed 17" not in repr(events)
    assert "0.12" not in repr(events)


@pytest.mark.parametrize("value", [
    [], [None], [True], [17], [{}], [[]], ["valid", 17], [""], [" \n\t "],
    ["before\x00after"], ["x" * 2501], ["x" * 1249, "y" * 1250], ["x"] * 13,
])
def test_proposal_prose_normalization_rejects_invalid_types_and_bounds(value):
    draft = with_proposal()
    draft["next_runs"][0]["limitations"] = value
    original = deepcopy(draft)
    events = []
    runner = engine()
    runner.emit = events.append
    with pytest.raises(ValueError):
        runner.execute("finish_report", draft)
    assert draft == original
    assert events == []


@pytest.mark.parametrize("value", [
    None, True, 17, {}, "", " \n\t ", "x" * 1501, "before\x00after",
    "[]", '["a limitation"]', '{"text":"a limitation"}',
    '["A failed trial."], "A second limitation."]', '["unterminated', '{"text": "unfinished',
])
def test_report_prose_normalization_rejects_ambiguous_invalid_and_oversized_values(value):
    draft = report()
    draft["limitations"] = value
    with pytest.raises(ValueError):
        engine().execute("finish_report", draft)


def test_malformed_container_prose_is_rejected_actionably_without_modification():
    draft = report()
    draft["limitations"] = '["The pilot failed."], "No paired endpoint was measured."]'
    original = deepcopy(draft)
    with pytest.raises(ValueError, match="native JSON array of prose strings"):
        normalize_report_transport(draft)
    assert draft == original


def test_exact_text_limits_are_accepted_without_truncation():
    draft = with_proposal()
    draft["limitations"] = "x" * 1500
    draft["next_runs"][0]["limitations"] = ["a" * 1249, "b" * 1249]
    result = engine().execute("finish_report", draft)["report"]
    assert result["limitations"][0] == "x" * 1500
    assert result["next_runs"][0]["limitations"] == "a" * 1249 + "\n\n" + "b" * 1249


@pytest.mark.parametrize("field,value", [
    ("changes", ["one change"]), ("budget", 112), ("budget", ["112 episodes"]),
    ("controls", "local"), ("metrics", "raw cost"), ("evidence_ids", "E001"),
    ("stop_rule", ["stop"]), ("title", ["title"]), ("hypothesis", ["hypothesis"]),
    ("extra", "not in the schema"),
])
def test_normalization_does_not_repair_any_other_proposal_field(field, value):
    draft = with_proposal()
    draft["next_runs"][0]["limitations"] = ["Development only."]
    draft["next_runs"][0][field] = value
    with pytest.raises(ValueError):
        engine().execute("finish_report", draft)


def test_normalization_does_not_repair_claim_shapes_or_wrong_citations():
    draft = with_proposal()
    draft["next_runs"][0]["limitations"] = ["Development only."]
    draft["claims"][0]["evidence"][0]["quote"] = "0.99"
    result = engine().execute("finish_report", draft)["report"]
    assert result["claims"][0]["verification"]["status"] == "unsupported"
    assert draft["claims"][0]["text"] not in result["answer"]
    assert result["verification"]["semantic_review_required"] is True
    for field in ("text", "evidence"):
        malformed = deepcopy(draft)
        malformed["claims"][0][field] = json.dumps(malformed["claims"][0][field]) if field == "evidence" else ["claim"]
        with pytest.raises(ValueError):
            engine().execute("finish_report", malformed)


def test_normalization_does_not_bypass_proposal_command_rejection():
    draft = with_proposal()
    draft["next_runs"][0]["limitations"] = ["Development only.", "python train.py"]
    with pytest.raises(ValueError, match="not commands"):
        engine().execute("finish_report", draft)
