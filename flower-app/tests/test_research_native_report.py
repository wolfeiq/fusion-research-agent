"""Native report tool arguments preserve deterministic citation enforcement."""

import json
from copy import deepcopy

import pytest

from fusion_agent.research.engine import REPORT_SCHEMA, TOOLS, ResearchEngine


def report():
    return {
        "answer": "Uncited model prose must not survive.",
        "claims": [{
            "id": "C001", "text": "The recorded endpoint cost was 0.12.",
            "kind": "observation", "confidence": "high", "evidence": [{
                "source_id": "E001", "quote": "0.12", "locator": "/cost",
            }],
        }],
        "next_runs": [], "limitations": ["No literature was retrieved."],
        "stop_reason": "Insufficient evidence for a wider claim.",
    }


def engine():
    source = {
        "id": "E001", "kind": "experiment", "content": '{"cost":0.12}',
        "title": "Saved test fixture", "uri": "results/test.json",
    }
    result = ResearchEngine([source], None, None, lambda _event: None)
    result.planned = result.assessed = True
    result.read_ids.add("E001")
    return result


def test_native_report_preserves_claim_validation_and_discards_uncited_prose():
    result = engine().execute("finish_report", {"report": report()})["report"]
    assert result["verification"]["traceable_claims"] == 1
    assert result["answer"] == "The recorded endpoint cost was 0.12. [E001]"
    assert result["verification"]["semantic_review_required"]


def test_flat_report_fields_have_identical_validation_to_saved_legacy_shapes():
    flat = engine().execute("finish_report", report())
    native = engine().execute("finish_report", {"report": report()})
    encoded = engine().execute("finish_report", {"report": json.dumps(report())})
    legacy = engine().execute("finish_report", {"report_json": json.dumps(report())})
    assert flat == native == encoded == legacy


def test_flat_report_does_not_accept_nested_serialization_or_extra_fields():
    draft = report()
    draft["claims"] = json.dumps(draft["claims"])
    with pytest.raises(ValueError, match="claims must be a list"):
        engine().execute("finish_report", draft)
    for extra in ({"report": report()}, {"report_json": json.dumps(report())}, {"extra": 1}):
        with pytest.raises(ValueError):
            engine().execute("finish_report", {**report(), **extra})


def test_flat_report_still_rejects_wrong_quotes_and_missing_required_fields():
    draft = report()
    draft["claims"][0]["evidence"][0]["quote"] = "0.99"
    result = engine().execute("finish_report", draft)["report"]
    assert result["claims"][0]["verification"]["status"] == "unsupported"
    assert draft["claims"][0]["text"] not in result["answer"]
    del draft["stop_reason"]
    with pytest.raises(ValueError):
        engine().execute("finish_report", draft)


def test_native_report_rejects_a_quote_that_does_not_occur_at_the_pointer():
    draft = report()
    draft["claims"][0]["evidence"][0]["quote"] = "0.99"
    result = engine().execute("finish_report", {"report": draft})["report"]
    assert result["claims"][0]["verification"]["status"] == "unsupported"
    assert draft["claims"][0]["text"] not in result["answer"]


@pytest.mark.parametrize("args", [
    {}, {"report": report(), "extra": True},
    {"report": report(), "report_json": json.dumps(report())},
    {"report_json": json.dumps(report()), "extra": True},
    {"report": []}, {"report": None},
])
def test_malformed_mixed_and_extra_top_level_arguments_are_rejected(args):
    with pytest.raises(ValueError):
        engine().execute("finish_report", args)


@pytest.mark.parametrize("mutation", [
    lambda draft: draft.update(extra="unadvertised"),
    lambda draft: draft["claims"][0].update(extra="unadvertised"),
    lambda draft: draft["claims"][0]["evidence"][0].update(extra="unadvertised"),
    lambda draft: draft["claims"][0].update(kind="fact"),
    lambda draft: draft["claims"][0].update(confidence="certain"),
    lambda draft: draft["claims"][0]["evidence"][0].update(locator="/missing"),
])
def test_nested_schema_and_citation_checks_remain_enforced(mutation):
    draft = report()
    mutation(draft)
    if draft["claims"][0]["evidence"][0]["locator"] == "/missing":
        result = engine().execute("finish_report", {"report": draft})["report"]
        assert result["claims"][0]["verification"]["status"] == "unsupported"
    else:
        with pytest.raises(ValueError):
            engine().execute("finish_report", {"report": draft})


def test_exact_legacy_shape_has_identical_validation_for_saved_transcripts():
    native = engine().execute("finish_report", {"report": report()})
    legacy = engine().execute("finish_report", {"report_json": json.dumps(report())})
    assert native == legacy
    with pytest.raises(ValueError):
        engine().execute("finish_report", {"report_json": json.dumps(report()) + " {}"})


def test_once_encoded_report_is_normalized_with_a_safe_audit_event():
    events = []
    runner = engine()
    runner.emit = events.append
    native = engine().execute("finish_report", {"report": report()})
    encoded = runner.execute("finish_report", {"report": json.dumps(report())})
    assert encoded == native
    assert len(events) == 1
    assert events[0]["kind"] == "report_transport_normalized"
    assert events[0]["decoded_layers"] == 1
    assert "0.12" not in repr(events)


def test_encoded_report_does_not_repair_wrong_quotations_or_skip_semantic_review():
    draft = report()
    draft["claims"][0]["evidence"][0]["quote"] = "0.99"
    result = engine().execute("finish_report", {"report": json.dumps(draft)})["report"]
    assert result["claims"][0]["verification"]["status"] == "unsupported"
    assert draft["claims"][0]["text"] not in result["answer"]
    assert result["verification"]["semantic_review_required"]


@pytest.mark.parametrize("encoded", [
    "{broken", json.dumps(report()) + " {}", json.dumps(report()) + " trailing prose",
    json.dumps(json.dumps(report())), "[]", "null", "42", "true",
    '{"unexpected": NaN}', '{"unexpected": Infinity}', '{"unexpected": -Infinity}',
    " " * 50001, json.dumps({"answer": "x" * 50001}),
], ids=[
    "malformed", "trailing-object", "trailing-prose", "double-encoded",
    "array", "null", "number", "boolean", "nan", "infinity", "negative-infinity",
    "oversize-whitespace", "oversize-object",
])
def test_encoded_report_rejects_malformed_trailing_nested_nonfinite_and_oversized(encoded):
    with pytest.raises(ValueError):
        engine().execute("finish_report", {"report": encoded})


@pytest.mark.parametrize("args", [
    {"report": json.dumps(report()), "extra": "ignored"},
    {"report": json.dumps(report()), "report_json": json.dumps(report())},
])
def test_encoded_report_still_requires_an_exact_single_argument_envelope(args):
    with pytest.raises(ValueError, match="exactly one report"):
        engine().execute("finish_report", args)


def test_encoded_report_preserves_closed_nested_schema_validation():
    draft = report()
    draft["claims"][0]["evidence"][0]["invented"] = "field"
    with pytest.raises(ValueError, match="requires exactly these fields"):
        engine().execute("finish_report", {"report": json.dumps(draft)})


def test_native_report_retains_total_size_and_read_source_requirements():
    draft = report()
    draft["answer"] = "x" * 50_001
    with pytest.raises(ValueError, match="50000-character"):
        engine().execute("finish_report", {"report": draft})
    unread = engine()
    unread.read_ids.clear()
    with pytest.raises(ValueError, match="Read selected"):
        unread.execute("finish_report", {"report": report()})


def test_advertised_schema_uses_native_objects_with_closed_bounded_children():
    from fusion_agent.research.verification import MAX_EVIDENCE_PER_CLAIM

    tool = next(tool for tool in TOOLS if tool["function"]["name"] == "finish_report")
    parameters = tool["function"]["parameters"]
    assert parameters == REPORT_SCHEMA
    assert set(parameters["required"]) == {
        "answer", "claims", "next_runs", "limitations", "stop_reason",
    }
    assert "report" not in parameters["properties"]
    assert "report_json" not in json.dumps(tool)
    assert parameters["properties"]["claims"]["items"]["properties"]["evidence"]["maxItems"] == MAX_EVIDENCE_PER_CLAIM == 16

    def check(schema):
        if schema["type"] == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
            for child in schema["properties"].values():
                check(child)
        elif schema["type"] == "array":
            assert 0 <= schema["minItems"] <= schema["maxItems"]
            check(schema["items"])
        else:
            assert schema["type"] == "string"
            assert "enum" in schema or 0 <= schema["minLength"] <= schema["maxLength"]

    check(deepcopy(parameters))
