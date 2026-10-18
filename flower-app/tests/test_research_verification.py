"""Deterministic research evidence checks never make a network request."""

import json
from copy import deepcopy

import pytest

from fusion_agent.research.verification import (
    render_answer,
    resolve_json_pointer,
    validate_report,
)


def source(identifier="E001", kind="experiment", **overrides):
    result = {
        "id": identifier,
        "kind": kind,
        "title": "Recorded results",
        "uri": "https://example.org/paper" if kind == "web" else "results/run.json",
        "content": json.dumps({"metrics": {"cost": 0.12}, "seeds": [17, 29]}),
        "sha256": "a" * 64,
        "metadata": {},
    }
    result.update(overrides)
    return result


def report():
    return {
        "answer": "The model's uncited explanation must never survive validation.",
        "claims": [
            {
                "id": "C001",
                "text": "The recorded endpoint cost was 0.12.",
                "kind": "observation",
                "confidence": "high",
                "evidence": [
                    {"source_id": "E001", "quote": "0.12", "locator": "/metrics/cost"}
                ],
            }
        ],
        "next_runs": [
            {
                "title": "Repeat the comparison",
                "hypothesis": "Stability may vary by seed.",
                "changes": "Repeat the frozen design with three development seeds.",
                "controls": ["Local", "Reset-only"],
                "metrics": ["Endpoint cost"],
                "budget": "18 evaluation episodes",
                "stop_rule": "Stop after the fixed budget.",
                "evidence_ids": ["E001"],
                "limitations": "Development evidence only.",
            }
        ],
        "limitations": ["Small sample."],
        "stop_reason": "The search budget was reached.",
    }


def claim_status(result):
    return result["claims"][0]["verification"]["status"]


def test_valid_local_claim_is_traceable_not_verified_and_summary_is_rebuilt():
    original = report()
    frozen = deepcopy(original)
    result = validate_report(original, [source()])
    assert original == frozen
    assert claim_status(result) == "traceable"
    assert result["answer"] == "The recorded endpoint cost was 0.12. [E001]"
    assert result["verification"]["semantic_review_required"] is True
    assert result["verification"]["mode"] == "structural_traceability_only"
    assert result["next_runs"][0]["status"] == "proposal_only"


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [{"source_id": "W999", "quote": "0.12", "locator": "/metrics/cost"}],
        [{"source_id": "E001", "quote": "0.13", "locator": "/metrics/cost"}],
        [{"source_id": "E001", "quote": "0.12", "locator": "/seeds/0"}],
        [{"source_id": "E001", "quote": "0.12", "locator": "/absent"}],
        [{"source_id": "E001", "quote": "0.12", "locator": ""}],
        [{"source_id": "E001", "quote": "", "locator": "/metrics/cost"}],
    ],
)
def test_unsupported_claims_cannot_supply_the_answer(evidence):
    data = report()
    data["claims"][0]["evidence"] = evidence
    result = validate_report(data, [source()])
    assert claim_status(result) == "unsupported"
    assert result["claims"][0]["confidence"] == "low"
    assert result["answer"] == "Insufficient traceable evidence for a factual answer."


@pytest.mark.parametrize(
    "kind,uri,content",
    [
        ("search", "https://example.org/paper", "A result of 0.12."),
        ("web", "https://example.org/paper", ""),
        ("web", "javascript:alert(1)", "A result of 0.12."),
        ("web", "file:///private/paper", "A result of 0.12."),
        ("web", "https://user:secret@example.org/paper", "A result of 0.12."),
    ],
)
def test_unread_sources_and_fake_urls_are_not_evidence(kind, uri, content):
    result = validate_report(report(), [source(kind=kind, uri=uri, content=content)])
    assert claim_status(result) == "unsupported"


def test_model_cannot_inject_its_own_citation_url():
    data = report()
    data["claims"][0]["evidence"][0]["url"] = "https://invented.example/paper"
    with pytest.raises(ValueError, match="exactly"):
        validate_report(data, [source()])


def test_web_quotes_allow_whitespace_differences_but_not_case_or_omissions():
    data = report()
    data["claims"][0]["evidence"][0].update(quote="A measured result", locator="")
    web = source(kind="web", content="A  measured\n result was reported.")
    assert claim_status(validate_report(data, [web])) == "traceable"
    data["claims"][0]["evidence"][0]["quote"] = "a measured result"
    assert claim_status(validate_report(data, [web])) == "unsupported"


def test_aggregate_quote_cap_counts_distinct_quotes_and_redacts_overflow():
    words = [f"word{index}" for index in range(30)]
    data = report()
    data["claims"][0]["evidence"][0].update(quote=" ".join(words[:15]), locator="")
    second = deepcopy(data["claims"][0])
    second["id"] = "C002"
    second["evidence"][0]["quote"] = " ".join(words[15:26])
    data["claims"].append(second)
    result = validate_report(data, [source(kind="web", content=" ".join(words))])
    assert all(
        item["verification"]["status"] == "unsupported" for item in result["claims"]
    )
    assert all(item["evidence"][0]["quote"] == "" for item in result["claims"])


def test_duplicate_quotes_do_not_exhaust_word_budget_twice():
    data = report()
    quote = " ".join(f"word{index}" for index in range(25))
    data["claims"][0]["evidence"][0].update(quote=quote, locator="")
    second = deepcopy(data["claims"][0])
    second["id"] = "C002"
    data["claims"].append(second)
    result = validate_report(data, [source(kind="web", content=quote)])
    assert all(
        item["verification"]["status"] == "traceable" for item in result["claims"]
    )


def test_matching_a_quote_cannot_establish_that_a_claim_is_true():
    data = report()
    data["claims"][0]["text"] = "The cost was not 0.12 and federation is proven."
    result = validate_report(data, [source()])
    assert claim_status(result) == "traceable"
    assert result["verification"]["semantic_review_required"] is True
    # The engine must downgrade this after semantic review and re-render.
    result["claims"][0]["verification"]["status"] = "unsupported"
    assert "federation is proven" not in render_answer(result["claims"])


def test_hypotheses_are_explicit_and_low_confidence():
    data = report()
    data["claims"][0].update(kind="hypothesis", confidence="high", evidence=[])
    result = validate_report(data, [source()])
    assert claim_status(result) == "hypothesis"
    assert result["claims"][0]["confidence"] == "low"
    assert "Hypothesis — not established:" in result["answer"]


def test_protocol_text_can_be_cited_without_json_pointer():
    data = report()
    data["claims"][0]["evidence"][0].update(quote="Three paired seeds", locator="")
    result = validate_report(
        data, [source(kind="protocol", content="Three paired seeds.")]
    )
    assert claim_status(result) == "traceable"


def test_json_pointer_resolves_escaped_keys_and_arrays():
    document = {"a/b": {"m~n": [None, {"value": 3}]}}
    assert resolve_json_pointer(document, "/a~1b/m~0n/1/value") == 3
    assert resolve_json_pointer(document, "") is document


@pytest.mark.parametrize(
    "pointer",
    ["no-slash", "/x~2y", "/values/01", "/values/-1", "/values/-", "/values/8"],
)
def test_bad_json_pointers_are_rejected(pointer):
    with pytest.raises(ValueError):
        resolve_json_pointer({"values": [1]}, pointer)


@pytest.mark.parametrize(
    "field,value",
    [
        ("controls", []),
        ("metrics", []),
        ("evidence_ids", []),
        ("evidence_ids", ["W999"]),
    ],
)
def test_proposals_require_controls_metrics_and_known_evidence(field, value):
    data = report()
    data["next_runs"][0][field] = value
    with pytest.raises(ValueError):
        validate_report(data, [source()])


@pytest.mark.parametrize(
    "command",
    [
        "python train.py",
        "kaggle kernels push -p run",
        "```sh\nrun\n```",
        "Start-Process training.exe",
        "Review first.\n$ rm -rf results",
    ],
)
def test_proposals_reject_executable_commands(command):
    data = report()
    data["next_runs"][0]["changes"] = command
    with pytest.raises(ValueError, match="not commands"):
        validate_report(data, [source()])


def test_proposals_cannot_supply_an_execution_field():
    data = report()
    data["next_runs"][0]["command"] = "run current experiment"
    with pytest.raises(ValueError, match="exactly"):
        validate_report(data, [source()])


@pytest.mark.parametrize(
    "mutation",
    ["many_claims", "many_proposals", "long_text", "duplicate_claim", "no_claims"],
)
def test_schema_bounds(mutation):
    data = report()
    if mutation == "many_claims":
        data["claims"] *= 13
    elif mutation == "many_proposals":
        data["next_runs"] *= 5
    elif mutation == "long_text":
        data["claims"][0]["text"] = "x" * 2001
    elif mutation == "duplicate_claim":
        data["claims"] *= 2
    else:
        data["claims"] = []
    with pytest.raises(ValueError):
        validate_report(data, [source()])


@pytest.mark.parametrize("field", ["kind", "confidence"])
def test_invalid_enum_type_raises_schema_error(field):
    data = report()
    data["claims"][0][field] = []
    with pytest.raises(ValueError):
        validate_report(data, [source()])
