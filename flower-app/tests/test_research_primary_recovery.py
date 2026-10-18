"""Bounded primary-source recovery after an ineligible author-mirror read."""
import copy
import hashlib
import json

import pytest

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.graph import _primary_recovery_state, build_research_graph
from fusion_agent.research.primary_sources import eligible_primary_source

MIRROR = "https://wangshusen.github.io/papers/AISTATS2022.pdf"
PRIMARY = "https://proceedings.mlr.press/v151/example22a.html"
TEXT = "Federated policies require evaluation across independent training seeds."


class Retrieval:
    def __init__(self, *, canonical=True, extract_fails=False):
        self.calls = []
        self.canonical = canonical
        self.extract_fails = extract_fails

    def search(self, query):
        self.calls.append(("search", query))
        count = sum(kind == "search" for kind, _ in self.calls)
        url = PRIMARY if self.canonical and count > 1 else MIRROR
        return {"results": [{"url": url, "title": "A public federated RL paper",
                             "content": "A discovery snippet, not evidence."}]}

    def extract(self, url):
        self.calls.append(("extract", url))
        if self.extract_fails and url == PRIMARY:
            return {"results": []}
        return {"results": [{"url": url, "raw_content": TEXT}]}


class RepairModel:
    def __init__(self, *, refuse=False):
        self.history = []
        self.read_ids = []
        self.refuse = refuse

    def complete(self, messages, tools, tool_choice="auto"):
        self.history.append(copy.deepcopy(messages))
        turn = len(self.history)
        if turn == 1:
            name, args = "search_literature", {
                "query": "federated reinforcement learning AISTATS",
                "reason": "Discover a relevant primary paper.",
            }
        elif turn in (2, 5) and not (turn == 5 and self.refuse):
            result = json.loads(messages[-1]["content"])
            name, args = "read_page", {
                "source_id": result["results"][0]["id"],
                "reason": "Read the discovered scientific paper.",
            }
            self.read_ids.append(args["source_id"])
        elif turn == 4 and not self.refuse:
            feedback = json.loads(messages[-1]["content"])
            assert feedback["primary_reference"]["status"] == "retry_required"
            assert "proceedings.mlr.press" in feedback["primary_reference"]["instruction"]
            name, args = "search_literature", {
                "query": "federated reinforcement learning AISTATS site:proceedings.mlr.press",
                "reason": "Locate the same paper at a recognized publication venue.",
            }
        else:
            name, args = "submit_findings", {
                "summary": "The extracted source offers a mechanism, not evidence of local efficacy.",
                "evidence_ids": [self.read_ids[-1]], "gaps": [],
            }
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"turn-{turn}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}, {}


def execute_branch(model, retrieval, **limits):
    events = []
    engine = ResearchEngine([], model, retrieval, events.append, **limits)
    engine.planned = True
    graph = build_research_graph(engine)
    findings = graph.nodes["literature_researcher"].bound.invoke({
        "question": "What does public research say?", "round": 1, "catalogue": [],
        "plan": {"literature_goal": "Find primary research on federated policy evaluation."},
    })["literature_findings"]
    return engine, findings, events


def test_mirror_finish_is_rejected_then_agent_selects_and_reads_canonical_source():
    model, retrieval = RepairModel(), Retrieval()
    engine, findings, events = execute_branch(model, retrieval)
    assert engine.model_calls == 6  # Existing role limit, not extra unbounded turns.
    assert engine.searches == engine.extracts == 2
    assert findings["evidence_ids"] == ["W002"]
    assert eligible_primary_source(engine.sources["W002"])
    assert not eligible_primary_source(engine.sources["W001"])
    assert sum(event["kind"] == "primary_reference_recovery" for event in events) == 1
    assert not any(event["kind"] == "primary_reference_unresolved" for event in events)
    mirror_feedback = json.loads(model.history[2][-1]["content"])["primary_reference"]
    assert mirror_feedback["recognized_publication_location"] is False
    assert mirror_feedback["eligible_extracted_reference"] is False
    assert "author mirror" in mirror_feedback["next_action"]
    # Eligibility is routing metadata, not a mutation of the stored source text/hash.
    assert "primary_reference" not in engine.sources["W001"]
    assert engine.sources["W001"]["sha256"] == hashlib.sha256(TEXT.encode()).hexdigest()


def test_noncompliant_early_finish_stops_with_server_owned_gap_without_more_retrieval():
    model, retrieval = RepairModel(refuse=True), Retrieval()
    engine, findings, events = execute_branch(model, retrieval)
    assert engine.model_calls <= 6
    assert engine.searches == engine.extracts == 1
    assert findings["primary_reference_stop"]["status"] == "unresolved"
    assert "budget" in findings["primary_reference_stop"]["reason"]
    assert any("No eligible extracted primary" in gap for gap in findings["gaps"])
    assert any(event["kind"] == "primary_reference_unresolved" for event in events)
    assert not any(eligible_primary_source(s) for s in engine.sources.values())


def test_canonical_extraction_failure_is_retained_without_fabricating_eligible_evidence():
    model, retrieval = RepairModel(), Retrieval(extract_fails=True)
    engine, findings, events = execute_branch(model, retrieval)
    assert engine.model_calls == 6
    assert engine.extracts == 2
    assert findings["primary_reference_stop"]["status"] == "unresolved"
    assert any("No eligible extracted primary" in gap for gap in findings["gaps"])
    assert engine.sources["W002"]["kind"] == "search"
    assert not eligible_primary_source(engine.sources["W002"])
    assert any("could not extract" in event.get("message", "") for event in events)


@pytest.mark.parametrize("changes,reason", [
    ({"max_turns": 7}, "role_or_reserved_model_budget_cannot_complete_primary_recovery"),
    ({"max_calls": 3}, "tool_budget_reserved_for_report"),
    ({"searches": 6}, "search_budget_exhausted"),
    ({"extracts": 10}, "extraction_budget_exhausted"),
])
def test_recovery_never_overrides_global_or_provider_budgets(changes, reason):
    engine = ResearchEngine([], None, None, lambda event: None)
    for key, value in changes.items():
        setattr(engine, key, value)
    state = _primary_recovery_state(engine, set(), 2)
    assert state["status"] == "unresolved"
    assert state["reason"] == reason


def test_discovered_primary_can_be_read_in_one_remaining_turn_without_search_budget():
    engine = ResearchEngine([{"id": "W1", "kind": "search", "uri": PRIMARY}],
                            None, None, lambda event: None)
    engine.searches = 6
    state = _primary_recovery_state(engine, set(), 4)
    assert state["status"] == "retry_required"
    assert state["eligible_primary_source_ids"] == ["W1"]
    assert _primary_recovery_state(engine, set(), 5)["status"] == "unresolved"


def test_hash_bound_primary_already_personally_read_allows_early_finish():
    item = {"id": "W1", "kind": "web", "uri": PRIMARY, "content": TEXT,
            "sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
            "metadata": {"status": "extracted"}}
    engine = ResearchEngine([item], None, None, lambda event: None)
    assert _primary_recovery_state(engine, {"W1"}, 0)["status"] == "satisfied"
    item["sha256"] = "0" * 64
    assert _primary_recovery_state(engine, {"W1"}, 0)["status"] == "retry_required"
