"""Offline regression harness tests; no credentials or provider calls."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_research_citation_scope.py"
SPEC = importlib.util.spec_from_file_location("verify_research_citation_scope", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    candidate = {"configuration": "candidate", "seed": 17, "round": 1, "arm": "clipped",
                 "role": "deposition_sac", "clipped_source_counts": {"q0": 2, "q1": 2},
                 "clipped_sources": {"q0": ["donor_cool", "donor_warm"], "q1": ["donor_cool", "donor_warm"]},
                 "clipping_factors": {"donor_cool": {"q0": .5, "q1": .5},
                                      "donor_warm": {"q0": .6, "q1": .6},
                                      "cold_target": {"q0": 1.0, "q1": 1.0}}}
    document = {"round_aggregation_diagnostics": {"records": [{}, {}, {}, candidate]},
                "paired_difference": -.001, "absolute_candidate": .099}
    local = json.dumps(document)
    web = "Synthetic unit-test source. Centralized training clips gradients before adding calibrated noise. This sentence describes centralized sample privacy."
    sources = [{"id": "E001", "kind": "experiment", "uri": "synthetic.json", "content": local,
                "title": "Synthetic fixture", "metadata": {"synthetic": True}},
               {"id": "W004", "kind": "web", "uri": "https://example.org/synthetic",
                "content": web, "title": "Synthetic fixture", "metadata": {"synthetic": True}}]
    for source in sources:
        source["sha256"] = hashlib.sha256(source["content"].encode()).hexdigest()
    claims = [
        {"id": "C1", "text": "Candidate SAC clipped both donors; baseline and TD3 did not clip.",
         "kind": "observation", "confidence": "high", "evidence": [{"source_id": "E001",
          "locator": "/round_aggregation_diagnostics/records/3", "quote": json.dumps(candidate, sort_keys=True)}]},
        {"id": "C2", "text": "Difference is -0.001 and candidate absolute cost is 0.099.",
         "kind": "observation", "confidence": "high", "evidence": [{"source_id": "E001",
          "locator": "/paired_difference", "quote": "-0.001"}]},
        {"id": "C4", "text": "This passage establishes client-level federated privacy.",
         "kind": "literature", "confidence": "medium", "evidence": [{"source_id": "W004",
          "locator": "", "quote": "clips gradients before adding calibrated noise"}]},
    ]
    run = {"id": probe.RUN_ID, "status": "completed", "sources": sources,
           "report": {"claims": claims}}
    path = tmp_path / "run.json"
    raw = json.dumps(run).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(probe, "RUN_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(probe, "SOURCE_HASHES", {source["id"]: source["sha256"] for source in sources})
    return path


class FakeModel:
    def __init__(self, *, accepts=False, raw_accept=False, reject_control=False, mutate=None):
        self.calls = []
        self.accepts, self.raw_accept = accepts, raw_accept
        self.reject_control, self.mutate = reject_control, mutate

    def complete(self, messages, tools, tool_choice="auto"):
        self.calls.append(deepcopy(messages))
        assert [tool["function"]["name"] for tool in tools] == ["review_claims"]
        data = {"decision": "accept" if self.accepts or self.raw_accept else "revise",
                "verdicts": [{"id": identifier,
                              "verdict": ("supported" if (identifier == "CONTROL" and not self.reject_control)
                                          or self.accepts else "uncertain"),
                              "all_clauses_supported": ((identifier == "CONTROL" and not self.reject_control) or self.accepts),
                              "scope_matches": ((identifier == "CONTROL" and not self.reject_control) or self.accepts),
                              "unsupported_clauses": ([] if (identifier == "CONTROL" and not self.reject_control)
                                                      or self.accepts else ["Synthetic uncovered clause."]),
                              "reason": "Synthetic verdict for offline harness checks."}
                             for identifier in ("CONTROL", "C1", "C2", "C4")],
                "gaps": ["Repair exact citation coverage and scientific scope."], "proposal_verdicts": [],
                "rubric": {"citation_support": 1, "experimental_fairness": 1,
                           "actionability": 1, "scope_honesty": 1}}
        if self.mutate:
            self.mutate()
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": "judge1", "type": "function", "function": {
                "name": "review_claims", "arguments": json.dumps(data)}}]}, {"total_tokens": 40}


def test_exactly_one_review_detects_scope_gaps_and_retains_correct_control(saved_run):
    model = FakeModel()
    result = probe.run_challenge(saved_run, model)
    assert result["passed"] and all(result["checks"].values())
    assert result["model_calls"] == len(model.calls) == 1
    payload = json.loads(model.calls[0][1]["content"])
    assert "expected" not in payload
    assert [claim["id"] for claim in payload["claims"]] == ["CONTROL", "C1", "C2", "C4"]
    assert all(claim["verification"]["status"] == "traceable" for claim in result["traceability_validated_input"]["claims"])
    assert all("content" not in source for source in result["sources"])
    run = json.loads(saved_run.read_bytes())
    original = {claim["id"]: claim for claim in run["report"]["claims"]}
    for claim in result["traceability_validated_input"]["claims"][1:]:
        assert claim["text"] == original[claim["id"]]["text"]
        for actual, expected in zip(claim["evidence"], original[claim["id"]]["evidence"], strict=True):
            assert {key: actual[key] for key in expected} == expected


@pytest.mark.parametrize("option", [{"accepts": True}, {"raw_accept": True}, {"reject_control": True}])
def test_guarded_nonaccept_alone_or_blanket_rejection_cannot_pass(saved_run, option):
    result = probe.run_challenge(saved_run, FakeModel(**option))
    assert not result["passed"]
    if option.get("raw_accept"):
        assert result["reviewed_report"]["judge"]["decision"] == "revise"
        assert not result["checks"]["raw_model_decision_not_accept"]


def test_modified_run_fails_before_any_model_call(saved_run):
    saved_run.write_bytes(saved_run.read_bytes() + b" ")
    model = FakeModel()
    with pytest.raises(ValueError, match="Run bytes"):
        probe.run_challenge(saved_run, model)
    assert model.calls == []


def test_modified_captured_source_hash_is_rejected_even_when_run_hash_matches(saved_run, monkeypatch):
    run = json.loads(saved_run.read_bytes())
    run["sources"][0]["content"] += " "
    raw = json.dumps(run).encode()
    saved_run.write_bytes(raw)
    monkeypatch.setattr(probe, "RUN_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="source content hash"):
        probe.run_challenge(saved_run, FakeModel())


def test_mutation_during_review_is_reported_and_never_accepted(saved_run):
    result = probe.run_challenge(saved_run, FakeModel(mutate=lambda: saved_run.write_bytes(saved_run.read_bytes() + b" ")))
    assert not result["passed"] and not result["checks"]["run_bytes_unchanged"]
