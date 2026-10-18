"""Exercise real LangGraph scheduling with injected, offline provider doubles."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager

import pytest

from fusion_agent.research.engine import ResearchEngine, offline_report
from fusion_agent.research.graph import (
    AGENTS,
    RESEARCHERS,
    _findings,
    _reply,
    build_research_graph,
)


def source(computed=False):
    data = {"status": "complete", "cost": 0.12}
    if computed:
        data.update(
            {
                "schema": "fusion-federation-dossier-v1",
                "manifest": {
                    "identity": "frozen-fixture",
                    "config": {
                        "heating_critic_coordinates": "physical",
                        "seeds": [17, 29],
                        "rounds": 2,
                        "episodes_per_round": 6,
                    },
                    "expected_budget": {"total_episodes": 100},
                },
                "jobs": [
                    {
                        "id": "uniform-seed-17",
                        "rounds": [
                            {
                                "round": 2,
                                "channels": {
                                    role: {
                                        "source_norms": {
                                            "donor_a": {"q0": 1.0, "q1": 1.0},
                                            "donor_b": {"q0": 3.0, "q1": 3.0},
                                        },
                                        "thresholds": {"q0": 1.0, "q1": 1.0},
                                        "weights": {"donor_a": 0.5, "donor_b": 0.5},
                                    }
                                    for role in ("heating_td3", "deposition_sac")
                                },
                            }
                        ],
                    }
                ],
            }
        )
    content = json.dumps(data)
    return {
        "id": "E001",
        "kind": "experiment",
        "title": "Small heating comparison",
        "uri": "results/summary.json",
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "metadata": {},
    }


def report(web_id="W002", computed=False, clip_id="D001", trial_id="D002"):
    result = {
        "answer": "This uncited prose must not survive.",
        "claims": [
            {
                "id": "C1",
                "text": "The saved endpoint cost is 0.12.",
                "kind": "observation",
                "confidence": "high",
                "evidence": [
                    {"source_id": "E001", "quote": "0.12", "locator": "/cost"}
                ],
            },
            {
                "id": "C2",
                "text": "The extracted article says clipping limits large updates.",
                "kind": "literature",
                "confidence": "medium",
                "evidence": [
                    {
                        "source_id": web_id,
                        "quote": "Clipping can limit large updates.",
                        "locator": "",
                    }
                ],
            },
        ],
        "next_runs": [
            {
                "title": "Paired clipping comparison",
                "hypothesis": "Bounded deltas may reduce instability.",
                "changes": "Compare the frozen local, reset-only, uniform and clipped variants.",
                "controls": ["Local", "Reset-only", "Uniform"],
                "metrics": ["Endpoint cost", "Failures"],
                "budget": "Three development seeds with six target shots each.",
                "stop_rule": "Stop after the frozen shot budget or invalid physics.",
                "evidence_ids": ["E001", web_id],
                "limitations": "Does not establish a federation advantage.",
            }
        ],
        "limitations": ["Development evidence only."],
        "stop_reason": "Bounded evidence collection completed.",
    }
    if computed:
        result["claims"].append(
            {
                "id": "C3",
                "text": "The computed diagnostic bound is 0.5; efficacy is untested.",
                "kind": "observation",
                "confidence": "high",
                "evidence": [
                    {
                        "source_id": clip_id,
                        "quote": "0.5",
                        "locator": "/roles/heating_td3/branches/q0/candidate_mean_norm_upper_bound",
                    }
                ],
            }
        )
        result["next_runs"][0]["evidence_ids"].append(trial_id)
    return result


class FakeRetrieval:
    def __init__(self):
        self.calls = []

    def search(self, query):
        self.calls.append(("search", query))
        return {
            "results": [
                {
                    "url": "https://arxiv.org/abs/fixture",
                    "title": "Clipping study",
                    "content": "Discovery snippet.",
                }
            ],
            "usage": {"credits": 1},
        }

    def extract(self, url):
        self.calls.append(("extract", url))
        return {
            "results": [
                {"url": url, "raw_content": "Clipping can limit large updates."}
            ]
        }


class TeamModel:
    """Responses depend on role and its own transcript, not call arrival order."""

    def __init__(
        self, decisions=("accept",), *, forbidden_analyst_tool=False, computed=False
    ):
        self.decisions = decisions
        self.forbidden_analyst_tool = forbidden_analyst_tool
        self.computed = computed
        self.web_id = "W002"
        self.clip_id, self.trial_id = "D001", "D002"
        self.counts = Counter()
        self.histories = defaultdict(list)
        self.available_tools = defaultdict(set)
        self.lock = threading.Lock()
        self.parallel_barrier = threading.Barrier(4)
        self.research_threads = set()

    def complete(self, messages, tools, tool_choice="auto"):
        first = messages[0]["content"]
        role = (
            first.split("\n", 1)[0].removeprefix("Agent role: ")
            if first.startswith("Agent role:")
            else "judge"
        )
        with self.lock:
            self.counts[role] += 1
            step = self.counts[role]
            self.histories[role].append(copy.deepcopy(messages))
            self.available_tools[role].update(
                tool["function"]["name"] for tool in tools
            )
        if role in RESEARCHERS and step == 1:
            with self.lock:
                self.research_threads.add(threading.get_ident())
            # This would fail if a coordinator merely ran named roles serially.
            self.parallel_barrier.wait(timeout=4)
        if role in RESEARCHERS and [tool["function"]["name"] for tool in tools] == ["submit_findings"]:
            assert tool_choice == "auto"
            terminal = json.loads(messages[-1]["content"])
            name, args = "submit_findings", {
                "summary": "The available snapshots support only scoped observations; remaining gaps are explicit.",
                "evidence_ids": terminal["eligible_evidence_ids"],
                "gaps": [],
            }
        elif role == "planner":
            name, args = (
                "plan_research",
                {
                    "questions": [
                        "What does the saved result support?",
                        "What evidence challenges transfer?",
                    ],
                    "td3_goal": "Inspect heating action coordinates and drift.",
                    "sac_goal": "Inspect deposition control and entropy learning.",
                    "federation_goal": "Inspect role-separated aggregation and controls.",
                    "literature_goal": "Find primary evidence about clipped federated updates.",
                    "reason": "Separate measurements from published mechanisms.",
                },
            )
        elif role == "federation_strategist" and self.computed:
            if step == 1:
                name, args = (
                    "inspect_federation",
                    {
                        "source_id": "E001",
                        "reason": "Inspect the captured coordinator state.",
                    },
                )
            elif step == 2:
                name, args = (
                    "compare_clipping",
                    {
                        "source_id": "E001",
                        "job_id": "uniform-seed-17",
                        "round": 2,
                        "threshold_scale": 0.5,
                        "reason": "Calculate bounded clipping diagnostics.",
                    },
                )
            elif step == 3:
                self.clip_id = json.loads(messages[-1]["content"])["id"]
                name, args = (
                    "draft_policy_trial",
                    {
                        "source_id": "E001",
                        "change": "heating_critic_coordinates",
                        "value": "normalized",
                        "reason": "Prepare a typed proposal without executing it.",
                    },
                )
            else:
                self.trial_id = json.loads(messages[-1]["content"])["id"]
                name, args = (
                    "submit_findings",
                    {
                        "summary": "Computed bound 0.5 and a proposal-only trial; no performance claim.",
                        "evidence_ids": ["E001", self.clip_id, self.trial_id],
                        "gaps": [],
                    },
                )
        elif role in {"td3_specialist", "sac_specialist", "federation_strategist"}:
            forbidden = self.forbidden_analyst_tool and role == "td3_specialist"
            if forbidden and step == 1:
                name, args = "launch_experiment", {"command": "do not run"}
            elif (step - 1 - int(forbidden)) % 2 == 0:
                name, args = (
                    "read_experiment",
                    {"source_id": "E001", "reason": "Read the endpoint evidence."},
                )
            else:
                name, args = (
                    "submit_findings",
                    {
                        "summary": f"{role}: the saved cost is 0.12; causality is unresolved.",
                        "evidence_ids": ["E001"],
                        "gaps": [],
                    },
                )
        elif role == "literature_researcher":
            phase = (step - 1) % 3
            if phase == 0:
                name, args = (
                    "search_literature",
                    {
                        "query": "federated reinforcement learning clipped updates",
                        "reason": "Find a primary mechanism reference.",
                    },
                )
            elif phase == 1:
                search_result = json.loads(messages[-1]["content"])
                self.web_id = search_result["results"][0]["id"]
                name, args = (
                    "read_page",
                    {
                        "source_id": self.web_id,
                        "reason": "Read the article before citing it.",
                    },
                )
            else:
                name, args = (
                    "submit_findings",
                    {
                        "summary": "Clipping can limit large updates; this does not prove efficacy here.",
                        "evidence_ids": [self.web_id],
                        "gaps": [],
                    },
                )
        elif role == "writer":
            if step % 2:
                name, args = (
                    "assess_evidence",
                    {
                        "enough": True,
                        "gaps": [],
                        "reason": "Both local and extracted web evidence are present.",
                    },
                )
            else:
                name, args = (
                    "finish_report",
                    {
                        "report_json": json.dumps(
                            report(
                                self.web_id, self.computed, self.clip_id, self.trial_id
                            )
                        )
                    },
                )
                if json.loads(messages[1]["content"]).get("phase") == "writer_reflection":
                    args = json.loads(args["report_json"])
        elif role == "judge":
            decision = self.decisions[min(step - 1, len(self.decisions) - 1)]
            name, args = (
                "review_claims",
                {
                    "verdicts": [
                        {
                            "id": "C1",
                            "verdict": "supported",
                            "reason": "The pointed result matches the scoped claim.",
                        },
                        {
                            "id": "C2",
                            "verdict": "supported",
                            "reason": "The extracted passage supports this limited claim.",
                        },
                    ],
                    "decision": decision,
                    "gaps": ["Check another mechanism reference."]
                    if decision == "revise"
                    else [],
                    "proposal_verdicts": [
                        {
                            "index": 0,
                            "verdict": "adequate",
                            "reason": "Fixed budget and explicit controls.",
                        }
                    ],
                    "rubric": {
                        "citation_support": 2,
                        "experimental_fairness": 2,
                        "actionability": 2,
                        "scope_honesty": 2,
                    },
                },
            )
        else:
            raise AssertionError("Unexpected agent context: " + role)
        if role == "judge" and self.computed:
            args["verdicts"].append(
                {
                    "id": "C3",
                    "verdict": "supported",
                    "reason": "The computed source records the scoped bound; no efficacy claim.",
                }
            )
        if role == "judge":
            for verdict in args["verdicts"]:
                supported = verdict["verdict"] == "supported"
                verdict.update(all_clauses_supported=supported, scope_matches=supported,
                               unsupported_clauses=[] if supported else [verdict["reason"]])
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"{role}_{step}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def setup_engine(decisions=("accept",), **kwargs):
    events = []
    model = TeamModel(decisions, **kwargs)
    retrieval = FakeRetrieval()
    engine = ResearchEngine([source(model.computed)], model, retrieval, events.append)
    return engine, model, retrieval, events


def test_actual_graph_has_seven_nodes_and_native_parallel_join():
    engine, _, _, _ = setup_engine()
    diagram = build_research_graph(engine).get_graph()
    assert set(AGENTS).issubset(diagram.nodes)
    edges = {(edge.source, edge.target) for edge in diagram.edges}
    assert len(AGENTS) == 7
    for role in RESEARCHERS:
        assert ("planner", role) in edges
        assert (role, "writer") in edges
    assert ("judge", "planner") in edges


def test_full_graph_accepts_flat_report_tool_arguments(monkeypatch):
    engine, model, _, _ = setup_engine()
    original = model.complete
    flat_calls = []

    def flat_writer(messages, tools, tool_choice="auto"):
        message, usage = original(messages, tools, tool_choice)
        if messages[0]["content"].startswith("Agent role: writer\n"):
            tool = next(t for t in tools if t["function"]["name"] == "finish_report")
            assert set(tool["function"]["parameters"]["properties"]) == {
                "answer", "claims", "next_runs", "limitations", "stop_reason",
            }
            assert "direct arguments answer, claims" in messages[0]["content"]
            call = message["tool_calls"][0]
            if call["function"]["name"] == "finish_report":
                draft = json.loads(json.loads(call["function"]["arguments"])["report_json"])
                call["function"]["arguments"] = json.dumps(draft)
                flat_calls.append(draft)
        return message, usage

    monkeypatch.setattr(model, "complete", flat_writer)
    result = engine.run("Check a flat report with exact citations.")
    assert len(flat_calls) == 1
    assert result["orchestration"]["status"] == "accepted"
    assert "[E001]" in result["answer"]


def test_writer_resolves_exact_citation_and_large_value_hint_within_six_turns(monkeypatch):
    engine, model, _, events = setup_engine()
    data = json.loads(engine.sources["E001"]["content"])
    data["large"] = {"cost": 0.12, "padding": "x" * 2100}
    engine.sources["E001"]["content"] = json.dumps(data)
    engine.sources["E001"]["sha256"] = hashlib.sha256(json.dumps(data).encode()).hexdigest()
    original = model.complete
    writer_calls = []
    resolved_citation = None

    def citation_writer(messages, tools, tool_choice="auto"):
        nonlocal resolved_citation
        if not messages[0]["content"].startswith("Agent role: writer\n"):
            return original(messages, tools, tool_choice)
        writer_calls.append(copy.deepcopy(messages))
        step = len(writer_calls)
        expected = ({"assess_evidence", "cite_experiment_value", "cite_web_excerpt", "finish_report"}
                    if step <= 4 else {"assess_evidence"} if step == 5 else {"finish_report"})
        assert {tool["function"]["name"] for tool in tools} == expected
        assert tool_choice == "auto"
        if step == 1:
            name, args = "cite_experiment_value", {
                "source_id": "E001", "path": ["large"], "reason": "Locate the saved result.",
            }
        elif step in (2, 3, 4):
            if step == 2:
                hint = json.loads(messages[-1]["content"])["citation_path_hint"]
                assert hint["child_keys"] == ["cost", "padding"]
            name, args = "cite_experiment_value", {
                "source_id": "E001", "path": ["large", "cost"],
                "reason": "Resolve an exact compact cost citation.",
            }
            if step > 2:
                resolved_citation = json.loads(messages[-1]["content"])
        elif step == 5:
            name, args = "assess_evidence", {
                "enough": True, "gaps": [], "reason": "Local and extracted sources are available.",
            }
        else:
            assert step == 6  # The writer may use its sixth bounded turn.
            citation = resolved_citation
            assert citation == {"source_id": "E001", "locator": "/large/cost", "quote": "0.12"}
            name, args = "finish_report", report(model.web_id)
            args["claims"][0]["evidence"] = [citation]
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"writer_citation_{step}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}, {"total_tokens": 1}

    monkeypatch.setattr(model, "complete", citation_writer)
    result = engine.run("Use an exact compact citation for the recorded cost.")
    assert result["orchestration"]["status"] == "accepted"
    assert len(writer_calls) == 6
    assert result["claims"][0]["evidence"][0]["locator"] == "/large/cost"
    assert sum(event["kind"] == "cite_experiment_value" for event in events) == 3
    assert engine.max_turns == 28 and engine.max_calls == 48


def test_writer_terminal_turns_retain_resolved_claims_and_report_unresolved_evidence(monkeypatch):
    engine, model, _, events = setup_engine(decisions=("insufficient",))
    original = model.complete
    writer_calls = 0
    resolved_citation = None

    def bounded_writer(messages, tools, tool_choice="auto"):
        nonlocal writer_calls, resolved_citation
        if not messages[0]["content"].startswith("Agent role: writer\n"):
            return original(messages, tools, tool_choice)
        writer_calls += 1
        names = {tool["function"]["name"] for tool in tools}
        calls = []
        if writer_calls <= 4:
            assert "cite_experiment_value" in names
            if writer_calls == 2:
                resolved_citation = json.loads(messages[-1]["content"])
            name, args = "cite_experiment_value", {
                "source_id": "E001", "path": ["cost" if writer_calls == 1 else "missing"],
                "reason": "Resolve selected measurements.",
            }
        elif writer_calls == 5:
            assert names == {"assess_evidence"} and tool_choice == "auto"
            # Execution must reject a tool even when the model ignores its available schema.
            calls.append({"id": "unavailable_lookup", "type": "function", "function": {
                "name": "cite_experiment_value", "arguments": json.dumps({
                    "source_id": "E001", "path": ["cost"], "reason": "Try another lookup.",
                }),
            }})
            name, args = "assess_evidence", {
                "enough": False, "gaps": ["The literature quotation remains unresolved."],
                "reason": "Only the recorded cost has resolved evidence.",
            }
        else:
            assert writer_calls == 6 and names == {"finish_report"} and tool_choice == "auto"
            args = report(model.web_id)
            args["claims"] = args["claims"][:1]
            args["claims"][0]["evidence"] = [resolved_citation]
            args["next_runs"] = []
            args["limitations"].append("The literature quotation remains unresolved; omitted that claim.")
            name = "finish_report"
        calls.append({"id": f"bounded_{writer_calls}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(args),
        }})
        return {"role": "assistant", "content": "", "tool_calls": calls}, {"total_tokens": 1}

    monkeypatch.setattr(model, "complete", bounded_writer)
    result = engine.run("Report the measured value and unresolved literature support honestly.")
    assert writer_calls == 6
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["judge"]["decision"] == "insufficient"
    assert result["evidence_sufficient"] is False
    assert [claim["id"] for claim in result["claims"]] == ["C1"]
    assert result["claims"][0]["verification"]["status"] == "traceable"
    assert any("literature quotation remains unresolved" in item for item in result["limitations"])
    assert len([event for event in events if event["kind"] == "writer_terminal_turn"]) == 2
    assert any(event["kind"] == "agent_tool_rejected" and "unavailable on the current writer turn" in event["message"] for event in events)


@pytest.mark.parametrize("gaps,description", [
    (["private detail"] * 13, "list of length 13"),
    ('["private detail"]', "str of length 18"),
])
def test_findings_gap_error_reports_safe_actual_type_and_count(gaps, description):
    engine, _, _, _ = setup_engine()
    with pytest.raises(ValueError) as raised:
        _findings({"summary": "Scoped findings.", "evidence_ids": [], "gaps": gaps}, engine, "td3_specialist", set())
    assert "remaining gaps must be a list of at most 12 strings" in str(raised.value)
    assert description in str(raised.value)
    assert "private detail" not in str(raised.value)


def test_full_graph_parallel_contexts_citations_judgment_and_proposals():
    engine, model, retrieval, events = setup_engine()
    result = engine.run("What should we test next?")
    assert result["orchestration"]["framework"] == "LangGraph"
    assert result["orchestration"]["rounds"] == 1
    assert result["orchestration"]["status"] == "accepted"
    assert result["judge"]["decision"] == "accept"
    assert result["evidence_sufficient"] is True
    assert result["next_runs"][0]["status"] == "proposal_only"
    assert len(model.research_threads) == 4
    assert set(model.counts) == set(AGENTS)
    assert len(retrieval.calls) == 2
    assert "uncited prose" not in result["answer"]
    assert "[E001]" in result["answer"] and f"[{model.web_id}]" in result["answer"]
    for role in ("planner", *RESEARCHERS, "writer"):
        assert model.histories[role][0][0]["content"].startswith(
            f"Agent role: {role}\n"
        )
        assert len(model.histories[role][0]) == 2
    assignments = {
        json.loads(model.histories[role][0][1]["content"])["assignment"]
        for role in RESEARCHERS
    }
    assert len(assignments) == 4
    writer_payload = json.loads(model.histories["writer"][0][1]["content"])
    for key in ("td3_findings", "sac_findings", "federation_findings"):
        assert writer_payload[key]["evidence_ids"] == ["E001"]
    assert writer_payload["literature_findings"]["evidence_ids"] == [model.web_id]
    completed = {
        event["agent"]: index
        for index, event in enumerate(events)
        if event["kind"] == "agent_complete"
    }
    writer_start = next(
        index
        for index, event in enumerate(events)
        if event["kind"] == "agent_start" and event["agent"] == "writer"
    )
    assert all(completed[role] < writer_start for role in RESEARCHERS)
    assert all("agent" in event for event in events if event["kind"] == "usage")
    assert result["usage"]["model_calls"] == 13
    assert model.available_tools["td3_specialist"] == {
        "read_experiment",
        "inspect_federation",
        "submit_findings",
    }
    assert (
        model.available_tools["sac_specialist"]
        == model.available_tools["td3_specialist"]
    )
    assert model.available_tools["federation_strategist"] == {
        "read_experiment",
        "inspect_federation",
        "compare_clipping",
        "draft_policy_trial",
        "submit_findings",
    }


def test_judge_revision_replans_with_gaps_then_accepts():
    engine, model, _, events = setup_engine(("revise", "accept"))
    engine.max_turns = 30  # Two complete rounds plus all four writer repair turns.
    result = engine.run("What should we test next?")
    assert result["orchestration"]["rounds"] == 2
    assert result["orchestration"]["status"] == "accepted"
    assert model.counts["planner"] == model.counts["judge"] == 2
    revised = json.loads(model.histories["planner"][1][1]["content"])
    assert revised["reviewer_gaps"] == ["Check another mechanism reference."]
    assert sum(event["kind"] == "agent_revision" for event in events) == 1
    assert result["usage"]["model_calls"] == 26
    assert result["usage"]["model_calls"] <= engine.max_turns


def test_judged_report_is_retained_after_one_reflection_when_research_retry_cannot_fit(monkeypatch):
    engine, model, _, events = setup_engine(("revise", "accept"))
    engine.max_turns = 24  # First round uses 13, leaving less than the 12-call reserve.
    original = model.complete

    def reject_one_claim(messages, tools, tool_choice="auto"):
        message, usage = original(messages, tools, tool_choice)
        if message["tool_calls"][0]["function"]["name"] == "review_claims":
            call = message["tool_calls"][0]
            args = json.loads(call["function"]["arguments"])
            args["verdicts"][1]["verdict"] = "unsupported"
            args["verdicts"][1]["reason"] = "The cited passage does not justify this interpretation."
            call["function"]["arguments"] = json.dumps(args)
        return message, usage

    monkeypatch.setattr(model, "complete", reject_one_claim)
    result = engine.run("Retain independently judged evidence if revision cannot fit.")
    assert result["orchestration"]["rounds"] == 1
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["orchestration"]["remaining_model_calls"] == 8
    assert result["judge"]["decision"] == "revise"
    assert result["evidence_sufficient"] is False
    assert result["claims"][1]["verification"]["status"] == "unsupported"
    assert result["claims"][1]["text"] not in result["answer"]
    assert model.counts["planner"] == 1 and model.counts["judge"] == 2
    assert result["orchestration"]["writer_reflections"] == 1
    assert any("12-call minimum" in item for item in result["limitations"])
    assert any(e["kind"] == "agent_revision_deferred" for e in events)
    assert not any(e["kind"] == "agent_revision" for e in events)


def test_writer_copies_one_contiguous_web_excerpt_into_judged_report(monkeypatch):
    engine, model, retrieval, events = setup_engine()
    original = model.complete

    def quote_writer(messages, tools, tool_choice="auto"):
        message, usage = original(messages, tools, tool_choice)
        if messages[0]["content"].startswith("Agent role: writer\n"):
            assert "cite_web_excerpt" in {tool["function"]["name"] for tool in tools}
            if model.counts["writer"] == 1:
                message["tool_calls"].append({
                    "id": "writer_web_quote", "type": "function", "function": {
                        "name": "cite_web_excerpt", "arguments": json.dumps({
                            "source_id": model.web_id, "anchor": "Clipping can limit",
                            "max_words": 5, "reason": "Copy one contiguous source excerpt.",
                        }),
                    },
                })
            else:
                citation = json.loads(messages[-1]["content"])
                assert citation["quote"] == "Clipping can limit large updates."
                call = message["tool_calls"][0]
                draft = json.loads(json.loads(call["function"]["arguments"])["report_json"])
                draft["claims"][1]["evidence"] = [citation]
                call["function"]["arguments"] = json.dumps(draft)
        return message, usage

    monkeypatch.setattr(model, "complete", quote_writer)
    result = engine.run("Cite a contiguous literature passage unchanged.")
    assert result["orchestration"]["status"] == "accepted"
    assert result["claims"][1]["evidence"][0]["locator"].startswith("chars:")
    assert len(retrieval.calls) == 2  # Helper did not retrieve anything again.
    assert any(event["kind"] == "cite_web_excerpt" for event in events)


def test_federation_specialist_uses_real_bounded_tools_and_cites_computed_outputs():
    engine, model, _, events = setup_engine(computed=True)
    original = copy.deepcopy(engine.sources["E001"])
    result = engine.run(
        "How should the next aggregation policy comparison be designed?"
    )
    assert result["orchestration"]["status"] == "accepted"
    assert engine.sources["E001"] == original
    clipping = engine.sources[model.clip_id]
    trial = engine.sources[model.trial_id]
    for derived in (clipping, trial):
        assert derived["kind"] == "experiment"
        assert (
            derived["sha256"] == hashlib.sha256(derived["content"].encode()).hexdigest()
        )
        assert derived["id"] in engine.read_ids
    diagnostic = json.loads(clipping["content"])
    assert (
        diagnostic["scope"]
        == "norm_only_counterfactual; no tensor reaggregation, rollout or predicted efficacy"
    )
    for role in ("heating_td3", "deposition_sac"):
        for branch in ("q0", "q1"):
            assert (
                diagnostic["roles"][role]["branches"][branch][
                    "candidate_mean_norm_upper_bound"
                ]
                == 0.5
            )
    proposal = json.loads(trial["content"])
    assert proposal["status"] == "proposal_only"
    assert proposal["applied"] is False and proposal["executable"] is False
    assert proposal["change"]["before"] == "physical"
    assert proposal["change"]["candidate"] == "normalized"
    assert {"local", "reset_only", "initial_base_pull"}.issubset(proposal["controls"])
    writer_payload = json.loads(model.histories["writer"][0][1]["content"])
    assert {model.clip_id, model.trial_id}.issubset(
        writer_payload["federation_findings"]["evidence_ids"]
    )
    assert any(
        claim["id"] == "C3" and claim["verification"]["status"] == "traceable"
        for claim in result["claims"]
    )
    assert f"[{model.clip_id}]" in result["answer"]
    assert {"inspect_federation", "compare_clipping", "draft_policy_trial"}.issubset(
        {
            event.get("tool")
            for event in events
            if event.get("agent") == "federation_strategist"
            and event["kind"] == "agent_tool"
        }
    )


def test_each_local_specialist_must_read_its_own_citations():
    engine, _, _, _ = setup_engine()
    engine.read_ids.add("E001")  # Another specialist's read is not enough.
    args = {"summary": "A scoped observation.", "evidence_ids": ["E001"], "gaps": []}
    with pytest.raises(ValueError, match="Each specialist must read"):
        _findings(args, engine, "td3_specialist", set())
    assert _findings(args, engine, "td3_specialist", {"E001"})["evidence_ids"] == [
        "E001"
    ]


def test_literature_cannot_cite_a_page_read_only_by_another_context():
    engine, model, _, _ = setup_engine()
    engine.run("Read a primary literature source.")
    args = {"summary": "A scoped literature finding.", "evidence_ids": [model.web_id], "gaps": []}
    with pytest.raises(ValueError, match="personally read"):
        _findings(args, engine, "literature_researcher", set())
    assert _findings(args, engine, "literature_researcher", {model.web_id})["evidence_ids"] == [model.web_id]


def test_repeated_source_reply_preserves_one_exact_snapshot_and_acknowledges_hash():
    original = source()
    saved = copy.deepcopy(original)
    messages, supplied = [], set()
    _reply(messages, {"id": "first"}, original, supplied)
    _reply(messages, {"id": "again"}, original, supplied)
    first, duplicate = (json.loads(message["content"]) for message in messages)
    assert first == saved
    assert "content" not in duplicate
    assert duplicate["content_reference"]["id"] == saved["id"]
    assert duplicate["content_reference"]["sha256"] == saved["sha256"]
    assert original == saved  # Canonical evidence was never edited.
    changed = {**original, "content": "A different snapshot", "sha256": "changed-hash"}
    _reply(messages, {"id": "changed"}, changed, supplied)
    assert json.loads(messages[-1]["content"])["content"] == changed["content"]


class ExtraDiscoveryRetrieval(FakeRetrieval):
    def search(self, query):
        result = super().search(query)
        result["results"].append({"url": "https://arxiv.org/abs/unread-fixture",
                                  "title": "Unread discovery", "content": "Not extracted."})
        return result


class BadFindingsModel(TeamModel):
    """Keep an actually read source while proposing an invalid evidence ID."""

    def __init__(self, *, role="literature_researcher", repair=False):
        super().__init__()
        self.target_role = role
        self.repair = repair
        self.terminal_requests = []

    def complete(self, messages, tools, tool_choice="auto"):
        first = messages[0]["content"]
        role = first.split("\n", 1)[0].removeprefix("Agent role: ") if first.startswith("Agent role:") else "judge"
        first_bad_turn = 3 if role == "literature_researcher" else 2
        if role != self.target_role or self.counts[role] + 1 < first_bad_turn:
            return super().complete(messages, tools, tool_choice)
        with self.lock:
            self.counts[role] += 1
            step = self.counts[role]
            self.histories[role].append(copy.deepcopy(messages))
            self.available_tools[role].update(tool["function"]["name"] for tool in tools)
        bad_id = "W003" if role == "literature_researcher" else "E999"
        name = "submit_findings"
        args = {"summary": "A deliberately invalid citation must be repaired before findings are accepted.",
                "evidence_ids": [bad_id], "gaps": []}
        if self.repair and step > first_bad_turn:
            correction = json.loads(messages[-1]["content"])
            args = {"summary": "A repaired scoped finding cites only personally read evidence.",
                    "evidence_ids": correction["eligible_evidence_ids"], "gaps": []}
        elif [tool["function"]["name"] for tool in tools] == ["submit_findings"]:
            self.terminal_requests.append((copy.deepcopy(tools), copy.deepcopy(tool_choice), copy.deepcopy(messages)))
            # Even a noncompliant provider cannot trigger further retrieval on
            # the terminal turn. The graph must enforce the narrowed tool set.
            name, args = "search_literature", {"query": "unwanted extra search", "reason": "A deliberately invalid terminal action."}
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"{role}_{step}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.parametrize("role,expected_id", [("literature_researcher", "W002"), ("td3_specialist", "E001")])
def test_incomplete_branch_retains_only_personally_read_sources(role, expected_id):
    events = []
    model, retrieval = BadFindingsModel(role=role), ExtraDiscoveryRetrieval()
    engine = ResearchEngine([source()], model, retrieval, events.append)
    result = engine.run("Preserve collected evidence even when a specialist fails to finish.")
    payload = json.loads(model.histories["writer"][0][1]["content"])
    key = "literature_findings" if role == "literature_researcher" else "td3_findings"
    assert payload[key]["evidence_ids"] == [expected_id]
    assert "Incomplete specialist assessment" in payload[key]["summary"]
    assert "no specialist conclusion is certified" in payload[key]["summary"]
    assert payload[key]["gaps"]
    assert expected_id in {item["id"] for item in payload["sources"]}
    assert "W003" not in {item["id"] for item in payload["sources"]}
    assert model.terminal_requests
    tools, choice, messages = model.terminal_requests[0]
    assert [tool["function"]["name"] for tool in tools] == ["submit_findings"]
    assert choice == "auto"
    assert json.loads(messages[-1]["content"])["eligible_evidence_ids"] == [expected_id]
    assert len([call for call in retrieval.calls if call[0] == "search"]) == 1
    assert any(event["kind"] == "agent_tool_rejected" and event.get("agent") == role
               and event.get("tool") == "search_literature" for event in events)
    assert result["orchestration"]["status"] == "accepted"  # Writer/judge inspect the real snapshots directly.


def test_bad_findings_ids_receive_actionable_eligible_ids_and_can_be_repaired():
    model, retrieval = BadFindingsModel(repair=True), ExtraDiscoveryRetrieval()
    engine = ResearchEngine([source()], model, retrieval, lambda event: None)
    result = engine.run("Repair a discovery-only citation without repeating retrieval.")
    correction = json.loads(model.histories["literature_researcher"][3][-1]["content"])
    assert correction["eligible_evidence_ids"] == ["W002"]
    assert "W003" not in correction["eligible_evidence_ids"]
    payload = json.loads(model.histories["writer"][0][1]["content"])
    assert payload["literature_findings"]["evidence_ids"] == ["W002"]
    assert "repaired scoped finding" in payload["literature_findings"]["summary"]
    assert model.counts["literature_researcher"] == 4
    assert not model.terminal_requests
    assert result["orchestration"]["status"] == "accepted"


def test_second_revision_request_stops_with_insufficient_evidence():
    engine, model, _, _ = setup_engine(("revise", "revise"))
    result = engine.run("What should we test next?")
    assert model.counts["judge"] == 2
    assert result["orchestration"]["rounds"] == 2
    assert result["orchestration"]["status"] == "insufficient_evidence"
    assert result["evidence_sufficient"] is False
    assert any("two-round" in item for item in result["limitations"])


def test_insufficient_judge_stops_without_claiming_completion():
    engine, model, _, _ = setup_engine(("insufficient",))
    result = engine.run("What should we test next?")
    assert model.counts["planner"] == model.counts["judge"] == 1
    assert result["evidence_sufficient"] is False
    assert result["orchestration"]["status"] == "insufficient_evidence"


def test_live_graph_explicitly_disables_third_party_tracing(monkeypatch):
    import langsmith

    original_context = langsmith.tracing_context
    enabled_values = []

    @contextmanager
    def record_context(**kwargs):
        enabled_values.append(kwargs.get("enabled"))
        with original_context(**kwargs):
            yield

    monkeypatch.setattr(langsmith, "tracing_context", record_context)
    engine, _, _, _ = setup_engine()
    result = engine.run("What should we test next?")
    assert result["orchestration"]["status"] == "accepted"
    assert enabled_values == [False]


def test_role_forbidden_tool_never_reaches_engine_execution():
    engine, model, _, events = setup_engine(forbidden_analyst_tool=True)
    original = engine.execute
    executed = []

    def execute(name, args):
        executed.append(name)
        return original(name, args)

    engine.execute = execute
    result = engine.run("What should we test next?")
    assert result["orchestration"]["status"] == "accepted"
    assert "launch_experiment" not in executed
    assert any(
        event.get("tool") == "launch_experiment"
        and event["kind"] == "agent_tool_rejected"
        for event in events
    )
    rejected_reply = json.loads(model.histories["td3_specialist"][1][-1]["content"])
    assert set(rejected_reply["allowed_tool_names"]) == {
        "read_experiment", "inspect_federation", "submit_findings"
    }


@pytest.mark.parametrize("role", ["planner", "td3_specialist", "writer"])
def test_oversized_batch_executes_nothing_and_recovers_within_existing_turns(monkeypatch, role):
    engine, model, _, events = setup_engine()
    original_complete, original_execute = model.complete, engine.execute
    injected = False
    checked_repair = False
    rejected_executions = []
    marker = "REJECTED_BATCH_SENTINEL"

    def complete(messages, tools, tool_choice="auto"):
        nonlocal injected, checked_repair
        current = messages[0]["content"].split("\n", 1)[0].removeprefix("Agent role: ")
        if current == role and not injected:
            injected = True
            if role == "planner":
                name, args = "plan_research", {
                    "questions": ["What is measured?"], "td3_goal": "Inspect heating.",
                    "sac_goal": "Inspect deposition.", "federation_goal": "Inspect aggregation.",
                    "literature_goal": "Find public evaluation guidance.", "reason": marker,
                }
            elif role == "writer":
                name, args = "assess_evidence", {"enough": True, "gaps": [], "reason": marker}
            else:
                name, args = "read_experiment", {"source_id": "E001", "reason": marker}
            assert name in {tool["function"]["name"] for tool in tools}
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"rejected_batch_{i}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args),
                }} for i in range(9)
            ]}, {"total_tokens": 1}
        if current == role and not checked_repair:
            checked_repair = True
            feedback = json.loads(messages[-1]["content"])
            assert messages[-1]["role"] == "user"
            assert feedback["actual_tool_call_count"] == 9 and feedback["maximum_tool_calls"] == 8
            assert "rejected_batch_" not in json.dumps(messages)
            assert not rejected_executions
        return original_complete(messages, tools, tool_choice)

    def execute(name, args):
        if args.get("reason") == marker:
            rejected_executions.append(name)
        return original_execute(name, args)

    monkeypatch.setattr(model, "complete", complete)
    monkeypatch.setattr(engine, "execute", execute)
    result = engine.run("Recover from a rejected batch without assuming any call ran.")
    assert injected and checked_repair and not rejected_executions
    assert result["orchestration"]["status"] == "accepted"
    assert model.counts["judge"] == 1
    rejection = next(event for event in events if event["kind"] == "agent_batch_rejected")
    assert rejection["actual_tool_call_count"] == 9 and rejection["maximum_tool_calls"] == 8
    assert rejection["agent"] == role
    assert engine.max_turns == 28 and engine.max_calls == 48


def test_role_prompts_resolve_global_search_scope_and_discourage_duplicate_reads():
    engine, model, _, _ = setup_engine()
    engine.run("Assess local evidence and relevant primary literature.")
    for role in ("td3_specialist", "sac_specialist", "federation_strategist"):
        prompt = model.histories[role][0][0]["content"]
        assert "The literature_researcher owns all web retrieval" in prompt
        assert "EITHER read_experiment OR inspect_federation" in prompt
        assert "do not call both for one source" in prompt
        expected = sorted(model.available_tools[role] - {"submit_findings"}) + ["submit_findings"]
        assert "Your exact permitted tool names are: " + ", ".join(expected) + "." in prompt
        assert "Prefer a summary of about 1,200 characters" in prompt
    literature_prompt = model.histories["literature_researcher"][0][0]["content"]
    assert "You are a LOCAL evidence specialist" not in literature_prompt
    assert "Your exact permitted tool names are: read_page, search_literature, submit_findings." in literature_prompt
    writer_prompt = model.histories["writer"][0][0]["content"]
    assert "smallest complete" in writer_prompt and "2000 characters" in writer_prompt
    assert "never reconstruct a subset" in writer_prompt
    assert "once per relevant measured value" not in writer_prompt
    # Concision is a prompt preference, not a new schema limit that could
    # discard important measurements or force another corrective model call.
    findings = _findings({"summary": "Measured details. " * 100, "evidence_ids": ["E001"], "gaps": []},
                         engine, "td3_specialist", {"E001"})
    assert len(findings["summary"]) > 1200


def test_literature_assignment_omits_local_catalog_question_and_artifact_identifiers(monkeypatch):
    engine, model, _, _ = setup_engine()
    trial_id = "43d55b69d02440b3a019cf9b64f6845a"
    engine.sources["E001"]["metadata"]["private_note"] = "PRIVATE_METADATA_SENTINEL"
    original = model.complete

    def contaminated_planner(messages, tools, tool_choice="auto"):
        message, usage = original(messages, tools, tool_choice)
        if messages[0]["content"].startswith("Agent role: planner\n"):
            call = message["tool_calls"][0]
            args = json.loads(call["function"]["arguments"])
            args["literature_goal"] = (
                f"Find primary literature on clipped federated critic updates for E001 "
                f"at results/summary.json from trial {trial_id}."
            )
            call["function"]["arguments"] = json.dumps(args)
        return message, usage

    monkeypatch.setattr(model, "complete", contaminated_planner)
    engine.run(f"PRIVATE_QUESTION_SENTINEL: assess trial {trial_id}.")
    payload = json.loads(model.histories["literature_researcher"][0][1]["content"])
    assert set(payload) == {"assignment"}
    assert "clipped federated critic updates" in payload["assignment"]
    serialized = json.dumps(payload)
    for private in ("E001", "results/summary.json", trial_id, "PRIVATE_METADATA_SENTINEL", "PRIVATE_QUESTION_SENTINEL"):
        assert private not in serialized
    for role in ("td3_specialist", "sac_specialist", "federation_strategist"):
        local = json.loads(model.histories[role][0][1]["content"])
        assert local["selected_evidence"][0]["id"] == "E001"
        assert local["selected_evidence"][0]["metadata"]["private_note"] == "PRIVATE_METADATA_SENTINEL"
    assert "separately read all local records" in model.histories["literature_researcher"][0][0]["content"]


def test_writer_receives_exact_citation_failure_details_and_can_repair(monkeypatch):
    engine, model, _, events = setup_engine()
    original = model.complete

    def bad_first_pointer(messages, tools, tool_choice="auto"):
        message, usage = original(messages, tools, tool_choice)
        if messages[0]["content"].startswith("Agent role: writer\n") and model.counts["writer"] == 2:
            tool_call = message["tool_calls"][0]
            args = json.loads(tool_call["function"]["arguments"])
            draft = json.loads(args["report_json"])
            draft["claims"][0]["evidence"][0]["locator"] = "/content/cost"
            tool_call["function"]["arguments"] = json.dumps({"report": draft})
        return message, usage

    monkeypatch.setattr(model, "complete", bad_first_pointer)
    result = engine.run("Check the exact source-relative citation before judging.")
    reply = json.loads(model.histories["writer"][2][-1]["content"])
    feedback = reply["citation_failures"]
    assert feedback["claims"][0]["evidence"][0]["source_id"] == "E001"
    assert feedback["claims"][0]["evidence"][0]["locator"] == "/content/cost"
    assert feedback["local_source_top_level_keys"]["E001"]["keys"] == ["status", "cost"]
    assert result["orchestration"]["status"] == "accepted"
    assert model.counts["writer"] == 4
    assert any(event["kind"] == "report_citation_validation" for event in events)


def test_findings_tool_advertises_existing_validation_limits():
    engine, model, _, _ = setup_engine()
    actual_complete = model.complete
    schemas = []

    def capture_complete(messages, tools, tool_choice="auto"):
        schemas.extend(copy.deepcopy(tool["function"]["parameters"])
                       for tool in tools if tool["function"]["name"] == "submit_findings")
        return actual_complete(messages, tools, tool_choice)

    model.complete = capture_complete
    engine.run("Keep the tool schema aligned with runtime evidence bounds.")
    assert schemas
    for schema in schemas:
        properties = schema["properties"]
        assert properties["summary"] == {"type": "string", "minLength": 1, "maxLength": 6000}
        assert properties["evidence_ids"]["maxItems"] == 24
        assert properties["evidence_ids"]["items"]["maxLength"] == 96
        assert properties["gaps"]["maxItems"] == 12
        assert properties["gaps"]["items"] == {"type": "string", "minLength": 1, "maxLength": 600}
        assert set(schema["required"]) == {"summary", "evidence_ids", "gaps"}
        assert schema["additionalProperties"] is False
    boundary = {"summary": "s" * 6000, "evidence_ids": ["E001"] * 24, "gaps": ["g" * 600] * 12}
    assert _findings(boundary, engine, "td3_specialist", {"E001"}) == boundary
    for field, value in (("summary", "s" * 6001), ("evidence_ids", ["E001"] * 25),
                         ("gaps", ["gap"] * 13), ("gaps", ["g" * 601])):
        with pytest.raises(ValueError):
            _findings({**boundary, field: value}, engine, "td3_specialist", {"E001"})


def test_offline_report_does_not_import_langgraph(monkeypatch):
    original_import = builtins.__import__

    def block_langgraph(name, *args, **kwargs):
        if name.startswith("langgraph"):
            raise AssertionError("Offline inspection must not import LangGraph")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_langgraph)
    result = offline_report([source()])
    assert "complete" in result["answer"]
    assert "[E001]" in result["answer"]


def test_missing_live_dependency_has_actionable_error(monkeypatch):
    engine, _, _, _ = setup_engine()
    original_import = builtins.__import__

    def block_langgraph(name, *args, **kwargs):
        if name.startswith("langgraph"):
            raise ImportError("Dependency intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_langgraph)
    with pytest.raises(RuntimeError, match="requires LangGraph"):
        build_research_graph(engine)
