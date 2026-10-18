"""The real graph selects evidence; even a gullible judge cannot author facts."""

import copy
import hashlib
import json

import pytest
from test_research_graph import FakeRetrieval, TeamModel

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.grounded_facts import build_grounded_facts
from fusion_agent.research.grounded_writer import (
    _draft,
    finalize_grounded,
    review_grounded,
)

pytest_plugins = ["test_research_grounded_facts"]


def web_source():
    content = "In centralized training, gradients are clipped for sample-level privacy."
    return {"id": "W002", "kind": "web", "title": "Synthetic source", "uri": "https://proceedings.mlr.press/v80/fujimoto18a.html",
            "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "metadata": {"status": "extracted"}}


def engine_for(source, model=None):
    engine = ResearchEngine([source, web_source()], model, None, lambda event: None)
    engine.read_ids.add(source["id"])
    return engine


def selection(source):
    cards = build_grounded_facts([source])
    return {"fact_ids": [next(c["id"] for c in cards if c["kind"] == "paired_raw_endpoint")],
            "literature": [{"source_id": "W002", "anchor": "In centralized training,", "max_words": 11}],
            "hypotheses": [{"text": "The absolute cost is 999 and this establishes client-level privacy.",
                            "evidence_ids": [source["id"], "W002"]}],
            "next_run_source_id": "", "gaps": []}


def call(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "bounded", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


class ApprovingJudge:
    def complete(self, messages, tools, tool_choice="auto"):
        return call("review_evidence_selection", {"decision": "accept", "gaps": [],
                    "rejected_hypothesis_indices": []}), {}


def test_bad_model_interpretation_never_gains_authority_even_when_judge_approves(source):
    engine = engine_for(source, ApprovingJudge())
    draft = _draft(engine, selection(source))
    reviewed = review_grounded(engine, {"draft": draft, "question": "What does the experiment establish?", "round": 1})["report"]
    assert reviewed["evidence_sufficient"]
    assert "999" not in reviewed["answer"]
    assert "client-level privacy" not in reviewed["answer"]
    assert "In centralized training" in reviewed["answer"]
    assert reviewed["model_interpretation"]["hypotheses"][0]["status"] == "untested"
    assert reviewed["grounded_publication"]["scientific_efficacy_established"] is False
    assert reviewed["grounded_publication"]["automatic_policy_changes_allowed"] is False
    assert all("999" not in claim["text"] for claim in reviewed["claims"])


def test_final_boundary_recompiles_instead_of_trusting_changed_report_text(source):
    engine = engine_for(source)
    draft = _draft(engine, selection(source))
    expected = copy.deepcopy(draft["grounded_publication"])
    draft.update(answer="Federation always wins: cost999", claims=[{"text": "Made up"}],
                 limitations=["No trajectories exist"], stop_reason="All claims verified", judge={"decision": "accept"})
    draft["grounded_publication"]["facts"][0]["text"] = "tampered"
    result = finalize_grounded(engine, draft)
    assert result["grounded_publication"] == expected
    assert result["answer"] == expected["answer"]
    assert result["claims"] == expected["facts"]
    assert "999" not in result["answer"]
    assert "No trajectories exist" not in result["limitations"]


@pytest.mark.parametrize("field,value", [("fact_ids", ["Fimaginary"]), ("hypotheses", "[]"),
                                        ("gaps", "none"), ("next_run_source_id", {})])
def test_selection_is_typed_and_cannot_add_evidence(source, field, value):
    args = selection(source)
    args[field] = value
    with pytest.raises((ValueError, TypeError)):
        _draft(engine_for(source), args)


def test_fact_text_and_literature_paraphrases_are_not_accepted_as_inputs(source):
    args = selection(source)
    args["claims"] = [{"text": "Federation cures cold start"}]
    with pytest.raises(ValueError, match="exactly"):
        _draft(engine_for(source), args)
    args = selection(source)
    args["literature"][0]["claim"] = "This supports client-level privacy"
    with pytest.raises(ValueError):
        _draft(engine_for(source), args)


def test_final_boundary_detects_source_mutation(source):
    engine = engine_for(source)
    draft = _draft(engine, selection(source))
    engine.sources[source["id"]]["content"] += " "
    with pytest.raises(ValueError, match="hash"):
        finalize_grounded(engine, draft)


def test_optional_hypothesis_unknown_sources_are_quarantined_not_guessed(source):
    engine = engine_for(source, ApprovingJudge())
    args = selection(source)
    args["hypotheses"][0]["evidence_ids"] = [args["fact_ids"][0], "W-imaginary"]
    draft = _draft(engine, args)
    assert draft["model_interpretation"]["hypotheses"] == []
    rejected = draft["model_interpretation"]["rejected_hypotheses"]
    assert len(rejected) == 1
    assert rejected[0]["status"] == "invalid_source_references"
    assert rejected[0]["evidence_ids"] == []
    assert rejected[0]["rejected_evidence_ids"] == args["hypotheses"][0]["evidence_ids"]
    reviewed = review_grounded(engine, {"draft": draft, "question": "What was measured?", "round": 1})["report"]
    result = finalize_grounded(engine, reviewed)
    assert result["evidence_sufficient"]
    assert "999" not in result["answer"]
    assert result["model_interpretation"]["rejected_hypotheses"] == rejected
    args["fact_ids"] = ["F-imaginary"]
    with pytest.raises(ValueError):
        _draft(engine, args)  # Optional prose rejection must never relax factual checks.


def current_source(saved, *, failed=False):
    content = json.loads(saved["content"])
    wrapper = {"schema": "fusion-advisor-shadow-evidence-v1", "trial_id": "current-trial",
               "status": "failed" if failed else "completed", "policy_activated": False,
               "comparison": None if failed else content,
               "round_aggregation_diagnostics": content["round_aggregation_diagnostics"]}
    text = json.dumps(wrapper)
    return {"id": "D001", "kind": "experiment", "content": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "metadata": {"adapter": "shadow_execution"}}


def test_current_run_cannot_be_substituted_with_historical_measured_values(source):
    engine = engine_for(source)
    current = current_source(source)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    cards = build_grounded_facts([current])
    args = selection(source)
    with pytest.raises(ValueError, match="trial_status"):
        _draft(engine, args)
    args["fact_ids"].append(next(c["id"] for c in cards if c["fact_type"] == "trial_status"))
    with pytest.raises(ValueError, match="paired_raw_endpoint"):
        _draft(engine, args)
    args["fact_ids"].append(next(c["id"] for c in cards if c["fact_type"] == "paired_raw_endpoint" and c["context"]["arm"] == "clipped"))
    with pytest.raises(ValueError, match="clipping"):
        _draft(engine, args)
    args["fact_ids"].append(next(c["id"] for c in cards if c["fact_type"] == "clipping"))
    answer = _draft(engine, args)["answer"]
    assert "Current research-run shadow" in answer and "Saved shadow" in answer


def test_failure_before_any_result_is_reportable_and_never_relabelled_success(source):
    engine = engine_for(source)
    current = current_source(source, failed=True)
    engine.sources[current["id"]] = current
    engine.read_ids.add(current["id"])
    args = selection(source)
    args["fact_ids"] = [next(c["id"] for c in build_grounded_facts(current) if c["fact_type"] == "trial_status")]
    args["next_run_source_id"] = current["id"]
    result = _draft(engine, args)
    assert "failed" in result["answer"]
    assert "0.1" not in result["answer"]
    assert result["grounded_publication"]["recommendation"]["request"] is None
    assert result["grounded_publication"]["recommendation"]["code"] == "diagnose_before_comparison"


def test_review_receives_exact_literature_context_and_preserves_failed_revision_diagnostic(source):
    class ContextJudge(ApprovingJudge):
        def complete(self, messages, tools, tool_choice="auto"):
            payload = json.loads(messages[1]["content"])
            packet = payload["captured_literature_context"]["claims"][0]["evidence"][0]
            assert "In centralized training" in packet["excerpts"][0]["text"]
            return super().complete(messages, tools, tool_choice)
    engine = engine_for(source, ContextJudge())
    args = selection(source)
    args["literature"][0].update(anchor="gradients are clipped", max_words=7)
    reviewed = review_grounded(engine, {"draft": _draft(engine, args), "question": "What is supported?", "round": 1})["report"]
    reviewed["orchestration"]["reflection_failed"] = True
    result = finalize_grounded(engine, reviewed)
    assert any("revision failed" in line for line in result["limitations"])
    assert result["grounded_publication"]["limitations"] == result["limitations"]


class GroundedTeam(TeamModel):
    def __init__(self):
        super().__init__()
        self.selections = 0
        self.reviews = 0

    def complete(self, messages, tools, tool_choice="auto"):
        names = {t["function"]["name"] for t in tools}
        if names == {"submit_evidence_report"}:
            self.selections += 1
            payload = json.loads(messages[1]["content"])
            cards = payload["fact_catalogue"]
            args = {"fact_ids": [next(c["id"] for c in cards if c["kind"] == "paired_raw_endpoint")],
                    "literature": [{"source_id": payload["web_sources"][0]["id"],
                                    "anchor": "Clipping can limit large updates.", "max_words": 6}],
                    "hypotheses": [], "next_run_source_id": "", "gaps": []}
            return call("submit_evidence_report", args), {"total_tokens": 10}
        if names == {"review_evidence_selection"}:
            self.reviews += 1
            return call("review_evidence_selection", {"decision": "accept", "gaps": [],
                        "rejected_hypothesis_indices": []}), {"total_tokens": 10}
        return super().complete(messages, tools, tool_choice)


class PrimaryRetrieval(FakeRetrieval):
    def search(self, query):
        result = super().search(query)
        result["results"][0]["url"] = "https://arxiv.org/abs/1802.09477"
        return result


def test_real_langgraph_uses_default_grounded_writer_and_read_only_judge(source):
    source.update(title="Synthetic shadow fixture", uri="results/summary.json")
    model, events = GroundedTeam(), []
    engine = ResearchEngine([source], model, PrimaryRetrieval(), events.append)
    report = engine.run("Inspect the recorded paired cost and a primary source.")
    assert engine.grounded_reporting is True
    assert model.selections == model.reviews == 1
    assert report["grounded_publication"]["status"] == "grounded"
    assert report["answer"] == report["grounded_publication"]["answer"]
    assert "0.12" not in report["answer"]  # Fabricated specialist summary is not factual authority.
    assert report["orchestration"]["framework"] == "LangGraph"
    assert report["judge"]["authority"] == "relevance_and_hypothesis_critique_only"
    assert any(event["kind"] == "grounded_draft" for event in events)
    assert any(event["kind"] == "grounded_review" for event in events)


def test_secondary_summary_cannot_satisfy_primary_slot_even_with_approving_judge(source):
    engine = engine_for(source, ApprovingJudge())
    engine.sources["W002"]["uri"] = "https://tinycto.tv/ml-atlas/literature/fujimoto-2018"
    args = selection(source)
    with pytest.raises(ValueError, match="approved primary-publication"):
        _draft(engine, args)
    args["literature"] = []
    report = review_grounded(engine, {"draft": _draft(engine, args), "question": "Require a primary paper.", "round": 1})["report"]
    result = finalize_grounded(engine, report)
    assert result["judge"]["decision"] == "accept"
    assert result["evidence_sufficient"] is False
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["grounded_publication"]["status"] == "insufficient_evidence"
    assert "No eligible extracted primary-publication" in result["stop_reason"]
    assert any("secondary summaries cannot" in line for line in result["limitations"])


def test_writer_offers_only_eligible_references_and_no_empty_enum(source):
    from fusion_agent.research.grounded_writer import write_grounded

    class InspectWriter:
        def __init__(self):
            self.expect_primary = False

        def complete(self, messages, tools, tool_choice="auto"):
            payload = json.loads(messages[1]["content"])
            schema = tools[0]["function"]["parameters"]["properties"]
            assert schema["fact_ids"]["maxItems"] == 12
            literature = schema["literature"]
            enum = literature["items"]["properties"]["source_id"].get("enum")
            args = selection(source)
            if self.expect_primary:
                assert enum == ["W002"] and len(payload["web_sources"]) == 1
            else:
                assert enum is None and literature["maxItems"] == 0
                assert payload["web_sources"] == []
                assert payload["primary_reference_gap"]
                args["literature"] = []
            return call("submit_evidence_report", args), {}

    model = InspectWriter()
    engine = engine_for(source, model)
    engine.sources["W002"]["uri"] = "https://example.org/summary"
    report = write_grounded(engine, {"question": "Primary evidence"})["draft"]
    assert report["grounded_publication"]["status"] == "insufficient_evidence"
    model.expect_primary = True
    engine.sources["W002"]["uri"] = "https://arxiv.org/abs/1802.09477"
    report = write_grounded(engine, {"question": "Primary evidence"})["draft"]
    assert report["grounded_publication"]["status"] == "grounded"


def test_final_literature_turn_can_extract_discovered_primary_without_extra_model_turn(source):
    class LateReader(GroundedTeam):
        def complete(self, messages, tools, tool_choice="auto"):
            if messages[0]["content"].startswith("Agent role: literature_researcher\n"):
                self.counts["literature_researcher"] += 1
                step = self.counts["literature_researcher"]
                if step == 1:
                    self.parallel_barrier.wait(timeout=4)
                if step < 6:
                    return call("search_literature", {"query": "TD3 primary methods paper", "reason": "Find the paper."}), {}
                assert step == 6
                assert {tool["function"]["name"] for tool in tools} == {"read_page", "submit_findings"}
                final = json.loads(messages[-1]["content"])
                assert final["eligible_primary_source_ids"]
                return call("read_page", {"source_id": final["eligible_primary_source_ids"][0], "reason": "Capture the discovered primary paper."}), {}
            return super().complete(messages, tools, tool_choice)

    source.update(title="Synthetic shadow fixture", uri="results/summary.json")
    model, retrieval, events = LateReader(), PrimaryRetrieval(), []
    report = ResearchEngine([source], model, retrieval, events.append).run("Inspect saved costs and a primary reference.")
    assert model.counts["literature_researcher"] == 6
    assert len([entry for entry in retrieval.calls if entry[0] == "extract"]) == 1
    assert report["grounded_publication"]["literature_excerpts"]
    literature_complete = [event for event in events if event["kind"] == "agent_complete" and event.get("agent") == "literature_researcher"]
    assert "unresolved gaps" in literature_complete[-1]["message"]
    assert literature_complete[-1]["evidence_ids"]
