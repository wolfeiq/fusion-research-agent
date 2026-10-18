"""Real LangGraph orchestration with offline model/retrieval/execution doubles.

The shadow executor is deliberately synthetic here. Real physics is verified by
the separately launched isolated worker, never by these provider contract tests.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_research_graph import FakeRetrieval, TeamModel, source

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.graph import AGENTS, RESEARCHERS, build_research_graph

SHADOW_ARGS = {
    "source_id": "E001", "change": "td3_actor_lr", "value": .00005,
    "reason": "Test one justified learning-rate change in an isolated pilot.",
}
SHADOW_RESULT = {
    "id": "fixture-shadow-only", "status": "completed",
    "comparison": {"status": "ready", "recommendation": "development_only",
                   "limitations": ["Synthetic executor response used by a protocol test."]},
}


class ShadowTeamModel(TeamModel):
    def __init__(self, *, duplicate=False, forbidden_role=None, read_late=False,
                 skip=False, decisions=("accept",)):
        super().__init__()
        self.decisions = decisions
        self.duplicate = duplicate
        self.forbidden_role = forbidden_role
        self.read_late = read_late
        self.shadow_id = None
        self.skip = skip
        self.decision_payloads = []
        self.parallel_tools = set()

    def complete(self, messages, tools, tool_choice="auto"):
        response, usage = super().complete(messages, tools, tool_choice)
        first = messages[0]["content"]
        role = first.split("\n", 1)[0].removeprefix("Agent role: ") if first.startswith("Agent role:") else "judge"
        step = self.counts[role]
        call = response["tool_calls"][0]["function"]
        payload = json.loads(messages[1]["content"])
        post_research = payload.get("phase") == "post_research_shadow_decision"
        stage_step = 1 + sum(message["role"] == "assistant" for message in messages)

        def choose(name, args):
            call.update(name=name, arguments=json.dumps(args))

        if role == "federation_strategist" and not post_research:
            self.parallel_tools.update(tool["function"]["name"] for tool in tools)

        if role == self.forbidden_role and step == 1:
            choose("test_shadow", SHADOW_ARGS)
        elif role == "federation_strategist" and post_research:
            if stage_step == 1:
                self.decision_payloads.append(copy.deepcopy(payload))
            if self.skip:
                choose("submit_findings", {
                    "summary": "The joined findings do not justify another pilot; preserve the existing evidence.",
                    "evidence_ids": [], "gaps": ["A new shadow trial is not warranted."],
                })
            elif self.read_late and stage_step == 1:
                choose("test_shadow", SHADOW_ARGS)
            elif stage_step == 1 or (self.read_late and stage_step == 2):
                choose("read_experiment", {"source_id": "E001", "reason": "Read this role's supporting evidence first."})
            elif (self.read_late and stage_step == 3) or (not self.read_late and stage_step == 2):
                choose("test_shadow", SHADOW_ARGS)
            elif stage_step >= 3:
                previous = json.loads(messages[-1]["content"])
                if previous.get("id", "").startswith("D"):
                    self.shadow_id = previous["id"]
                if self.duplicate and stage_step == 3:
                    choose("test_shadow", SHADOW_ARGS)
                else:
                    choose("submit_findings", {
                        "summary": "The bounded shadow comparison remains development-only.",
                        "evidence_ids": ["E001"] + ([self.shadow_id] if self.shadow_id else []),
                        "gaps": [] if self.shadow_id else ["Shadow execution was unavailable."],
                    })
        elif role == "federation_strategist":
            if stage_step == 1 or (self.forbidden_role == role and step == 2):
                choose("read_experiment", {"source_id": "E001", "reason": "Read the assigned immutable endpoint evidence."})
            else:
                choose("submit_findings", {"summary": "The saved endpoint cost is 0.12; the joined findings must justify a pilot.",
                                           "evidence_ids": ["E001"], "gaps": []})
        elif role in {"td3_specialist", "sac_specialist"} and self.forbidden_role == role:
            if step == 2:
                choose("read_experiment", {"source_id": "E001", "reason": "Read the assigned evidence after rejected authority."})
            else:
                choose("submit_findings", {"summary": "The recorded endpoint cost is 0.12.", "evidence_ids": ["E001"], "gaps": []})

        if role == "writer" and call["name"] == "finish_report" and self.shadow_id:
            args = json.loads(call["arguments"])
            report = json.loads(args["report_json"])
            report["claims"].append({
                "id": "C3", "text": "The shadow comparison recommends development-only follow-up.",
                "kind": "observation", "confidence": "medium",
                "evidence": [{"source_id": self.shadow_id, "quote": "development_only",
                              "locator": "/comparison/recommendation"}],
            })
            report["next_runs"][0]["evidence_ids"].append(self.shadow_id)
            choose("finish_report", {"report_json": json.dumps(report)})
        if role == "judge" and self.shadow_id:
            args = json.loads(call["arguments"])
            args["verdicts"].append({"id": "C3", "verdict": "supported",
                                     "reason": "The scoped recommendation matches the captured shadow result."})
            choose("review_claims", args)
        return response, usage


def setup(*, enabled=True, **kwargs):
    events, executions = [], []
    model = ShadowTeamModel(**kwargs)

    def executor(change, value):
        executions.append((change, value))
        return copy.deepcopy(SHADOW_RESULT)

    engine = ResearchEngine([source()], model, FakeRetrieval(), events.append,
                            shadow_executor=executor if enabled else None)
    return engine, model, events, executions


def test_shadow_result_flows_through_real_graph_to_writer_and_independent_judge():
    engine, model, events, executions = setup()
    original = copy.deepcopy(engine.sources["E001"])
    result = engine.run("Test a bounded heating change and suggest the next development run.")
    assert result["orchestration"]["framework"] == "LangGraph"
    assert result["orchestration"]["status"] == "accepted"
    assert executions == [("td3_actor_lr", .00005)]
    assert engine.sources["E001"] == original
    derived = engine.sources[model.shadow_id]
    assert derived["id"] in engine.read_ids
    assert derived["kind"] == "experiment"
    assert json.loads(derived["content"]) == SHADOW_RESULT
    assert derived["sha256"] == hashlib.sha256(derived["content"].encode()).hexdigest()
    assert derived["metadata"]["adapter"] == "shadow_execution"
    assert derived["metadata"]["input_sha256"] == original["sha256"]
    for role in ("writer", "judge"):
        payload = json.loads(model.histories[role][0][1]["content"])
        assert any(s["id"] == model.shadow_id and s["content"] == derived["content"] for s in payload["sources"])
    writer = json.loads(model.histories["writer"][0][1]["content"])
    assert writer["federation_findings"]["evidence_ids"] == ["E001", model.shadow_id]
    assert f"[{model.shadow_id}]" in result["answer"]
    claim = next(c for c in result["claims"] if c["id"] == "C3")
    assert claim["verification"]["status"] == "traceable"
    assert result["next_runs"][0]["status"] == "proposal_only"
    own_tools = [event.get("tool") for event in events
                 if event["kind"] == "agent_tool" and event.get("phase") == "post_research_shadow_decision"]
    assert own_tools[:2] == ["read_experiment", "test_shadow"]
    for role, available in model.available_tools.items():
        assert ("test_shadow" in available) == (role == "federation_strategist")
    assert "test_shadow" not in model.parallel_tools
    assert len(model.decision_payloads) == 1
    payload = model.decision_payloads[0]
    decision_history = next(history for history in model.histories["federation_strategist"]
                            if json.loads(history[1]["content"]).get("phase") == "post_research_shadow_decision")
    assert "Your exact permitted tool names are read_experiment, test_shadow and submit_findings." in decision_history[0]["content"]
    assert "call read_experiment for the supporting local source in this stage" in decision_history[0]["content"]
    decision_histories = [history for history in model.histories["federation_strategist"]
                          if json.loads(history[1]["content"]).get("phase") == "post_research_shadow_decision"]
    repeated_read = json.loads(decision_histories[1][-1]["content"])
    assert repeated_read["id"] == "E001" and "content" not in repeated_read
    assert repeated_read["content_reference"]["sha256"] == original["sha256"]
    initial_sources = json.loads(decision_histories[1][1]["content"])["sources"]
    assert next(source for source in initial_sources if source["id"] == "E001")["content"] == original["content"]
    fresh_shadow = json.loads(decision_histories[2][-1]["content"])
    assert fresh_shadow["id"] == model.shadow_id and fresh_shadow["content"] == derived["content"]
    for role in ("td3", "sac", "federation", "literature"):
        assert payload[f"{role}_findings"]["evidence_ids"]
    assert any(s["kind"] == "web" and "Clipping can limit" in s["content"] for s in payload["sources"])
    shadow_start = next(i for i, event in enumerate(events) if event["kind"] == "shadow_start")
    for role in RESEARCHERS:
        assert any(i < shadow_start and event["kind"] == "agent_complete" and event.get("agent") == role
                   for i, event in enumerate(events))
    assert result["usage"]["model_calls"] == 16


def test_second_shadow_attempt_is_rejected_without_second_execution():
    engine, model, events, executions = setup(duplicate=True)
    result = engine.run("Compare one isolated development change.")
    assert result["orchestration"]["status"] == "accepted"
    assert len(executions) == 1
    assert model.shadow_id in engine.sources
    assert any(event["kind"] == "agent_tool_rejected" and event.get("tool") == "test_shadow"
               and "already been used" in event["message"] for event in events)


def test_no_execution_capability_is_exposed_when_shadow_is_disabled():
    engine, model, events, executions = setup(enabled=False, forbidden_role="federation_strategist")
    result = engine.run("Read the existing evidence without a new experiment.")
    assert result["orchestration"]["status"] == "accepted"
    assert not executions and model.shadow_id is None
    assert not any("test_shadow" in available for available in model.available_tools.values())
    assert any(event["kind"] == "agent_tool_rejected" and event.get("tool") == "test_shadow"
               for event in events)


def test_another_agents_read_does_not_authorize_federation_execution():
    engine, model, events, executions = setup(read_late=True)
    # Other agents' reads must not satisfy the strategist's personal evidence
    # prerequisite, regardless of parallel scheduling order.
    engine.read_ids.add("E001")
    result = engine.run("Require the strategist to inspect its own evidence before testing.")
    assert result["orchestration"]["status"] == "accepted"
    assert executions == [("td3_actor_lr", .00005)]
    rejected = [event for event in events if event["kind"] == "agent_tool_rejected"
                and event.get("agent") == "federation_strategist"]
    assert rejected and rejected[0].get("tool") == "test_shadow"
    assert model.shadow_id in engine.sources
    correction = next(json.loads(message["content"])
                      for history in model.histories["federation_strategist"]
                      for message in history if message["role"] == "tool"
                      and "personally read its cited source in the decision stage" in message["content"])
    assert set(correction["allowed_tool_names"]) == {"read_experiment", "test_shadow", "submit_findings"}


@pytest.mark.parametrize("role", ["td3_specialist", "sac_specialist", "planner", "writer"])
def test_other_roles_cannot_acquire_shadow_authority_by_inventing_a_tool_call(role):
    engine, model, events, executions = setup(forbidden_role=role)
    result = engine.run("Read evidence and permit only the federation role's bounded pilot.")
    assert result["orchestration"]["status"] == "accepted"
    assert executions == [("td3_actor_lr", .00005)]
    assert "test_shadow" not in model.available_tools[role]
    assert any(event["kind"] == "agent_tool_rejected" and event.get("agent") == role
               for event in events)


def test_shadow_stage_is_a_native_fan_in_and_reuses_one_of_seven_roles():
    engine, _, _, _ = setup()
    diagram = build_research_graph(engine).get_graph()
    edges = {(edge.source, edge.target) for edge in diagram.edges}
    assert len(AGENTS) == 7
    assert "shadow_decision" in diagram.nodes
    for role in RESEARCHERS:
        assert (role, "shadow_decision") in edges
        assert (role, "writer") not in edges
    assert ("shadow_decision", "writer") in edges


def test_strategist_can_skip_without_a_forced_candidate_or_trial():
    engine, model, _, executions = setup(skip=True)
    result = engine.run("Assess whether another test would add useful evidence.")
    assert result["orchestration"]["status"] == "accepted"
    assert not executions and not engine.shadow_attempted
    assert len(model.decision_payloads) == 1
    writer = json.loads(model.histories["writer"][0][1]["content"])
    assert "do not justify" in writer["federation_findings"]["summary"]
    assert writer["federation_findings"]["evidence_ids"] == ["E001"]


def test_parallel_federation_role_cannot_launch_before_other_findings():
    engine, model, events, executions = setup(forbidden_role="federation_strategist")
    result = engine.run("Test only after the complete research findings are available.")
    assert result["orchestration"]["status"] == "accepted"
    assert executions == [("td3_actor_lr", .00005)]
    assert "test_shadow" not in model.parallel_tools
    assert any(event["kind"] == "agent_tool_rejected" and event.get("tool") == "test_shadow"
               and "phase" not in event for event in events)


def test_shadow_decision_reserves_writer_and_judge_calls():
    engine, model, events, executions = setup()
    engine.max_turns = 17  # Ten research calls plus six writer turns and a judge.
    result = engine.run("Complete the report even when there is no budget for a pilot decision.")
    assert result["orchestration"]["status"] == "accepted"
    assert not model.decision_payloads and not executions
    assert result["usage"]["model_calls"] == 13
    assert any(event["kind"] == "shadow_decision_skipped" for event in events)


def test_revision_retains_measured_shadow_evidence_without_a_second_decision():
    engine, model, events, executions = setup(decisions=("revise", "accept"))
    # The successful path needs 29 calls, plus four reserved writer correction
    # calls. This explicit budget exercises revision without another pilot.
    engine.max_turns = 33
    result = engine.run("Challenge the first draft and retain all pilot outcomes.")
    assert result["orchestration"]["rounds"] == 2
    assert result["orchestration"]["status"] == "accepted"
    assert executions == [("td3_actor_lr", .00005)]
    assert len(model.decision_payloads) == 1
    second_writer = json.loads(model.histories["writer"][2][1]["content"])
    assert model.shadow_id in second_writer["federation_findings"]["evidence_ids"]
    assert any(source["id"] == model.shadow_id for source in second_writer["sources"])
    assert any(event["kind"] == "shadow_decision_skipped" and "already been used" in event["message"]
               for event in events)
    assert result["usage"]["model_calls"] == 29
