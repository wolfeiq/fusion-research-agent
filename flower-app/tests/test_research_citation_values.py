"""Exact citation construction never substitutes for truth or provenance checks."""

import json

import pytest

from fusion_agent.research.citation_values import cite_experiment_value
from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.verification import resolve_json_pointer, validate_report


def source(data=None, *, kind="experiment"):
    return {
        "id": "E001", "kind": kind, "title": "Saved fixture", "uri": "fixture.json",
        "content": json.dumps(data if data is not None else {"cost": 0.12}),
    }


def runner(saved=None):
    events = []
    engine = ResearchEngine([saved or source()], None, None, events.append)
    engine.planned = True
    engine.read_ids.add("E001")
    return engine, events


def validated_claim(saved, citation):
    return validate_report({
        "answer": "", "claims": [{
            "id": "C1", "text": "A deliberately unjudged interpretation.",
            "kind": "observation", "confidence": "high", "evidence": [citation],
        }], "next_runs": [], "limitations": [], "stop_reason": "Citation transport test.",
    }, [saved])


@pytest.mark.parametrize("data,path,pointer,quote", [
    ({"a/b": {"c~d": [0.125]}}, ["a/b", "c~d", 0], "/a~1b/c~0d/0", "0.125"),
    ({"seeds": {"17": {"cost": 0.2}}}, ["seeds", "17", "cost"], "/seeds/17/cost", "0.2"),
    ({"seeds": [0.3]}, ["seeds", 0], "/seeds/0", "0.3"),
    ({"": None}, [""], "/", "null"),
    ({"value": {"z": 1, "a": 2}}, ["value"], "/value", '{"a": 2, "z": 1}'),
    ({"message": "escaped \"value\""}, ["message"], "/message", '"escaped \\"value\\""'),
])
def test_lookup_returns_exact_escaped_pointer_and_sorted_json_quote(data, path, pointer, quote):
    saved = source(data)
    citation = cite_experiment_value(saved, path)
    assert citation == {"source_id": "E001", "locator": pointer, "quote": quote}
    assert json.loads(quote) == resolve_json_pointer(data, pointer)
    report = validated_claim(saved, citation)
    assert report["claims"][0]["verification"]["status"] == "traceable"
    assert report["verification"]["semantic_review_required"]


@pytest.mark.parametrize("data,path", [
    ({"17": "value"}, [17]), ({"list": [1]}, ["list", "0"]),
    ({"list": [1]}, ["list", True]), ({"list": [1]}, ["list", -1]),
    ({"list": [1]}, ["list", 1]), ({"cost": 1}, ["missing"]),
    ({"cost": 1}, ["cost", "child"]), ({"cost": 1}, []),
    ({"cost": 1}, "/cost"), ({"cost": 1}, [None]),
    ({"cost": 1}, ["cost"] * 33), ({"cost": 1}, ["x" * 129]),
    ({"cost": 1}, ["x" * 128] * 4), ({"cost": 1}, ["\n"]),
])
def test_lookup_rejects_root_invalid_paths_and_implicit_key_index_coercion(data, path):
    with pytest.raises(ValueError):
        cite_experiment_value(source(data), path)


def test_large_object_returns_bounded_child_keys_without_partial_quote():
    saved = source({"summary": {"cost": 0.12, "long": "x" * 2100}})
    with pytest.raises(ValueError, match="2000-character") as raised:
        cite_experiment_value(saved, ["summary"])
    assert raised.value.citation_path_hint == {
        "source_id": "E001", "path": ["summary"], "locator": "/summary",
        "value_type": "object", "child_keys": ["cost", "long"], "omitted_children": 0,
    }
    citation = cite_experiment_value(saved, ["summary", "cost"])
    assert citation["quote"] == "0.12"


def test_large_array_returns_only_bounded_integer_child_indices():
    with pytest.raises(ValueError) as raised:
        cite_experiment_value(source({"values": ["x" * 100] * 100}), ["values"])
    hint = raised.value.citation_path_hint
    assert hint["child_indices"] == list(range(32))
    assert hint["omitted_children"] == 68


@pytest.mark.parametrize("path,expected_prefix,expected_children", [
    (["summary", "seed_cost"], ["summary"], {"child_keys": ["by_seed"]}),
    (["summary", "by_seed", "0"], ["summary", "by_seed"], {"child_indices": [0]}),
    (["summary", "by_seed", 0, "cost", "child"], ["summary", "by_seed", 0, "cost"], {"child_keys": []}),
    (["wrong_root"], [], {"child_keys": ["summary"]}),
])
def test_missing_path_returns_last_resolved_prefix_and_actionable_children(path, expected_prefix, expected_children):
    saved = source({"summary": {"by_seed": [{"cost": 0.12}]}})
    with pytest.raises(ValueError) as raised:
        cite_experiment_value(saved, path)
    hint = raised.value.citation_path_hint
    assert hint["path"] == expected_prefix
    assert hint["locator"] == ("/" + "/".join(map(str, expected_prefix)) if expected_prefix else "")
    assert all(hint[key] == value for key, value in expected_children.items())
    assert "quote" not in hint


def test_missing_key_hint_does_not_expose_unbounded_children_or_values():
    saved = source({str(i): "private value" for i in range(100)})
    with pytest.raises(ValueError) as raised:
        cite_experiment_value(saved, ["wrong"])
    hint = raised.value.citation_path_hint
    assert len(hint["child_keys"]) == 32 and hint["omitted_children"] == 68
    assert "private value" not in json.dumps(hint)


@pytest.mark.parametrize("kind", ["protocol", "web", "search"])
def test_engine_rejects_nonexperiment_even_if_already_read(kind):
    engine, _ = runner(source(kind=kind))
    with pytest.raises(ValueError, match="experiment JSON"):
        engine.execute("cite_experiment_value", {
            "source_id": "E001", "path": ["cost"], "reason": "Get an exact citation.",
        })


def test_engine_requires_read_source_and_exact_arguments_and_records_lookup():
    engine, events = runner()
    args = {"source_id": "E001", "path": ["cost"], "reason": "Get exact saved cost."}
    engine.read_ids.clear()
    with pytest.raises(ValueError, match="Read the selected"):
        engine.execute("cite_experiment_value", args)
    engine.read_ids.add("E001")
    with pytest.raises(ValueError, match="exactly"):
        engine.execute("cite_experiment_value", {**args, "quote": "invented"})
    assert engine.execute("cite_experiment_value", args) == {
        "source_id": "E001", "locator": "/cost", "quote": "0.12",
    }
    assert events[-1]["kind"] == "cite_experiment_value"
    assert "quote" not in events[-1]


def test_lookup_quote_does_not_weaken_later_mismatch_validation():
    saved = source()
    citation = cite_experiment_value(saved, ["cost"])
    citation["quote"] = "0.99"
    report = validated_claim(saved, citation)
    assert report["claims"][0]["verification"]["status"] == "unsupported"


@pytest.mark.parametrize("content", ['{"cost":NaN}', '{"cost":Infinity}', "not JSON"])
def test_nonfinite_or_malformed_source_is_rejected(content):
    saved = source()
    saved["content"] = content
    with pytest.raises(ValueError, match="valid finite JSON"):
        cite_experiment_value(saved, ["cost"])
