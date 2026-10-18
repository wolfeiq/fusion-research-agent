"""Adversarial check harness must not confuse matching quotations with truth."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_research_judge.py"
SPEC = importlib.util.spec_from_file_location("verify_research_judge", SCRIPT)
challenge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(challenge)


def source():
    return {
        "id": "E001", "kind": "experiment", "title": "Synthetic test fixture",
        "uri": "fixture.json", "sha256": "a" * 64,
        "metadata": {"synthetic_unit_test_only": True},
        "content": json.dumps({"candidate_vs_baseline_raw_endpoint": {"local": {
            "paired_training_seeds": 1, "by_seed": [{
                "seed": 17, "comparable": True, "mean_cost_difference": 0.03,
            }],
        }}}),
    }


class FakeModel:
    def __init__(self, *, rejects_false=True, decision="revise"):
        self.calls = []
        self.rejects_false = rejects_false
        self.decision = decision

    def complete(self, messages, tools, tool_choice):
        self.calls.append(deepcopy(messages))
        data = {
            "verdicts": [{
                "id": identifier,
                "verdict": ("supported" if identifier == "C001"
                            or not self.rejects_false else "unsupported"),
                "reason": "Synthetic verdict for harness validation.",
            } for identifier in ("C001", "C002", "C003")],
            "decision": self.decision, "gaps": ["Draft contains false conclusions"],
            "proposal_verdicts": [], "rubric": {
                "citation_support": 1, "experimental_fairness": 1,
                "actionability": 1, "scope_honesty": 0,
            },
        }
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": "review1", "type": "function", "function": {
                "name": "review_claims", "arguments": json.dumps(data),
            },
        }]}, {"total_tokens": 42}


class FakeCatalog:
    def __init__(self, *, changes=False):
        self.calls = []
        self.changes = changes

    def snapshot(self, record):
        self.calls.append(record)
        snapshot = source()
        if self.changes and len(self.calls) > 1:
            snapshot["sha256"] = "b" * 64
        return snapshot


def test_false_interpretations_match_true_scalar_but_invented_quote_fails():
    report, mismatch = challenge.build_challenge(source())
    assert all(c["verification"]["status"] == "traceable" for c in report["claims"])
    assert mismatch["passed"]
    assert mismatch["result"]["claims"][0]["verification"]["status"] == "unsupported"


def test_correctly_rejecting_judge_passes_one_call_and_checks_unchanged_source():
    model, catalog = FakeModel(), FakeCatalog()
    result = challenge.run_challenge("unused", model, catalog=catalog)
    assert result["passed"] and all(result["checks"].values())
    assert result["model_calls"] == len(model.calls) == 1
    assert catalog.calls == [challenge.RECORD, challenge.RECORD]
    assert result["usage"] == {"total_tokens": 42}
    payload = json.loads(model.calls[0][1]["content"])
    assert "expected" not in payload  # The model does not get the answer key.
    assert payload["sources"][0]["content"] == source()["content"]


def test_acceptance_guard_cannot_mask_a_model_that_accepts_false_claims():
    result = challenge.run_challenge(
        "unused", FakeModel(rejects_false=False, decision="accept"), catalog=FakeCatalog(),
    )
    assert not result["passed"]
    assert not result["checks"]["raw_model_decision_not_accept"]
    assert not result["checks"]["C002_false_claim_rejected_and_excluded"]


def test_source_change_between_review_and_final_check_fails():
    result = challenge.run_challenge("unused", FakeModel(), catalog=FakeCatalog(changes=True))
    assert not result["passed"]
    assert not result["checks"]["source_snapshot_and_provenance_unchanged"]


def test_different_experiment_shape_is_rejected_before_model_call():
    altered = source()
    data = json.loads(altered["content"])
    data["candidate_vs_baseline_raw_endpoint"]["local"]["by_seed"][0]["mean_cost_difference"] = -0.03
    altered["content"] = json.dumps(data)
    with pytest.raises(ValueError, match="frozen challenge"):
        challenge.build_challenge(altered)
