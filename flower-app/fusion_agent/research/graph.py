"""Seven distinct model contexts orchestrated by a bounded LangGraph StateGraph.

The four researchers run in parallel, using role-scoped tools and independent
conversation histories. A native fan-in waits for all four before writing. The
judge can request one new research round. If explicitly enabled, the federation
strategist receives the joined findings before deciding whether to request one
typed, isolated shadow pilot across the whole graph.
LangGraph is imported only when constructing a live graph, so offline evidence
inspection does not require it. See the official Graph API documentation:
https://docs.langchain.com/oss/python/langgraph/graph-api
"""

from __future__ import annotations

import copy
import json
from typing import TypedDict

from .retrieval_scope import public_literature_text
from .source_views import source_views

MAX_ROUNDS = 2
MAX_WRITER_REFLECTIONS = 1
REVIEW_CALL_RESERVE = 7  # Up to six writer turns plus one independent judge.
MAX_TOOL_BATCH = 8
MAX_WRITER_CITATION_BATCH = 24
CITATION_TOOLS = frozenset({"cite_experiment_value", "cite_web_excerpt"})
AGENTS = (
    "planner",
    "td3_specialist",
    "sac_specialist",
    "federation_strategist",
    "literature_researcher",
    "writer",
    "judge",
)
RESEARCHERS = AGENTS[1:5]
ROLE_GOALS = {
    "td3_specialist": "td3_goal",
    "sac_specialist": "sac_goal",
    "federation_strategist": "federation_goal",
    "literature_researcher": "literature_goal",
}
ROLE_FINDINGS = {
    "td3_specialist": "td3_findings",
    "sac_specialist": "sac_findings",
    "federation_strategist": "federation_findings",
    "literature_researcher": "literature_findings",
}


class ResearchState(TypedDict, total=False):
    question: str
    round: int
    gaps: list[str]
    plan: dict
    catalogue: list[dict]
    td3_findings: dict
    sac_findings: dict
    federation_findings: dict
    literature_findings: dict
    draft: dict
    report: dict
    writer_reflections: int
    reflection_draft_ready: bool


def _text(value, label, limit=1200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must contain 1–{limit} characters")
    return value.strip()


def _strings(value, label, limit=12, length=600):
    if not isinstance(value, list) or len(value) > limit:
        received = type(value).__name__
        if isinstance(value, (list, str, dict)):
            received += f" of length {len(value)}"
        raise ValueError(
            f"{label} must be a list of at most {limit} strings; received {received}. "
            "Return a native JSON array within the stated bounds."
        )
    return [_text(item, label, length) for item in value]


def _calls(message, *, limit=MAX_TOOL_BATCH):
    if not isinstance(message, dict):
        raise ValueError("Agent response must be an object")  # noqa: TRY004
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list) or len(calls) > limit:
        raise ValueError("Agent response exceeds the tool batch limit")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError("Agent returned an invalid tool call")  # noqa: TRY004
        _text(call.get("id"), "tool-call ID", 200)
        _text(call["function"].get("name"), "tool name", 100)
        if not isinstance(call["function"].get("arguments"), str):
            raise ValueError("Tool arguments must be JSON text")  # noqa: TRY004
    return calls


def _batch_limit(message, role, allowed_names):
    """Only the writer's currently available read-only citation helpers get 24."""
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if role == "writer" and isinstance(calls, list) and calls and all(
        isinstance(call, dict) and isinstance(call.get("function"), dict)
        and isinstance(call["function"].get("name"), str)
        and call["function"].get("name") in CITATION_TOOLS & set(allowed_names)
        for call in calls
    ):
        return MAX_WRITER_CITATION_BATCH
    return MAX_TOOL_BATCH


def _protected_claims(report):
    """Keep only canonical scientific fields from previously checked claims."""
    protected = {}
    for claim in report["claims"]:
        if (claim.get("verification", {}).get("status") == "traceable"
                and claim.get("review", {}).get("verdict") == "supported"):
            protected[claim["id"]] = {
                key: copy.deepcopy(claim[key]) for key in ("id", "text", "kind", "confidence")
            }
            protected[claim["id"]]["evidence"] = [
                {key: copy.deepcopy(item[key]) for key in ("source_id", "quote", "locator")}
                for item in claim["evidence"]
            ]
    return protected


def _preserve_protected_claims(args, protected):
    """Merge server-owned claims; validate all new/changed science downstream."""
    from .verification import MAX_CLAIMS, REPORT_FIELDS, decode_report_arguments

    draft, normalized = decode_report_arguments(args)
    if set(draft) != REPORT_FIELDS:
        raise ValueError("Reflection report requires exactly the five report fields")
    claims = draft["claims"]
    if not isinstance(claims, list) or len(claims) > MAX_CLAIMS:
        raise ValueError(f"Reflection claims must be a native array of at most {MAX_CLAIMS} items")
    seen, merged, replaced = set(), [], []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
            raise ValueError("Each reflection claim must have a string ID")  # noqa: TRY004 - public schema API
        identifier = claim["id"]
        if identifier in seen:
            raise ValueError("Claim IDs must be unique")
        seen.add(identifier)
        if identifier in protected:
            if claim != protected[identifier]:
                replaced.append(identifier)
            merged.append(copy.deepcopy(protected[identifier]))
        else:
            merged.append(copy.deepcopy(claim))
    restored = [identifier for identifier in protected if identifier not in seen]
    merged.extend(copy.deepcopy(protected[identifier]) for identifier in restored)
    merged_draft = {**draft, "claims": merged}
    # Preserve the existing envelope so engine validation retains its usual
    # explicit transport-normalization event for a once-encoded report string.
    if set(args) == {"report"}:
        merged_args = {"report": json.dumps(merged_draft, ensure_ascii=False, allow_nan=False) if normalized else merged_draft}
    elif set(args) == {"report_json"}:
        merged_args = {"report_json": json.dumps(merged_draft, ensure_ascii=False, allow_nan=False)}
    else:
        merged_args = merged_draft
    return merged_args, {"protected_claim_ids": list(protected),
                         "replaced_submitted_claim_ids": replaced,
                         "restored_omitted_claim_ids": restored}


def _messages(role, instructions, payload, system):
    return [
        {
            "role": "system",
            "content": (
                f"Agent role: {role}\n{system}\n\nYour independent assignment:\n{instructions}"
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _reply(messages, call, result, supplied_sources=None):
    if (supplied_sources is not None and isinstance(result, dict)
            and isinstance(result.get("id"), str)
            and isinstance(result.get("sha256"), str)
            and "content" in result):
        identity = (result["id"], result["sha256"])
        if identity in supplied_sources:
            # Keep the original immutable snapshot in context exactly once.
            # The read still executed and passed role-scoped provenance checks;
            # this is a transport acknowledgment, never a replacement source.
            result = {key: value for key, value in result.items() if key != "content"}
            result["content_reference"] = {
                "id": identity[0], "sha256": identity[1],
                "location": "Full source content was supplied earlier in this same agent context.",
            }
        else:
            supplied_sources.add(identity)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


def _findings(args, engine, role, role_read_ids):
    if not isinstance(args, dict) or set(args) != {"summary", "evidence_ids", "gaps"}:
        raise ValueError("Findings require summary, evidence_ids and gaps")
    result = {
        "summary": _text(args["summary"], "findings summary", 6000),
        "evidence_ids": _strings(args["evidence_ids"], "evidence IDs", 24, 96),
        "gaps": _strings(args["gaps"], "remaining gaps"),
    }
    for identifier in result["evidence_ids"]:
        source = engine.sources.get(identifier)
        if not source:
            raise ValueError("Findings cite an unknown source")
        if role != "literature_researcher":
            if (
                source["kind"] not in ("experiment", "protocol")
                or identifier not in engine.read_ids
                or identifier not in role_read_ids
            ):
                raise ValueError("Each specialist must read its cited local sources")
        elif source["kind"] != "web" or identifier not in role_read_ids:
            raise ValueError("The literature researcher must personally read an extracted page before citing it")
    return result


def build_research_graph(engine):
    """Build the actual parallel graph using an injected bounded engine facade."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise RuntimeError(
            "Live multi-agent research requires LangGraph; install the app's research dependencies"
        ) from None

    # Import at construction, avoiding an engine -> graph -> engine import cycle.
    from .engine import STR, SYSTEM, TOOLS, ModelCallBudgetReserved, function

    tools_by_name = {item["function"]["name"]: item for item in TOOLS}
    plan_tool = copy.deepcopy(tools_by_name["plan_research"])
    parameters = plan_tool["function"]["parameters"]
    parameters["properties"].update({name: STR for name in ROLE_GOALS.values()})
    parameters["required"] = [
        *parameters["required"],
        *ROLE_GOALS.values(),
    ]
    findings_tool = function(
        "submit_findings",
        "Return this researcher's evidence-backed findings and at most 12 remaining gaps.",
        {
            "summary": {"type": "string", "minLength": 1, "maxLength": 6000},
            "evidence_ids": {
                "type": "array", "maxItems": 24,
                "items": {"type": "string", "minLength": 1, "maxLength": 96},
            },
            "gaps": {
                "type": "array", "maxItems": 12,
                "items": {"type": "string", "minLength": 1, "maxLength": 600},
            },
        },
    )

    def eligible_evidence_ids(role, role_read_ids):
        kinds = {"web"} if role == "literature_researcher" else {"experiment", "protocol"}
        return sorted(identifier for identifier in role_read_ids
                      if identifier in engine.sources and engine.sources[identifier]["kind"] in kinds)

    def event(stage, role, message, state, **details):
        engine.event(stage, message, agent=role, round=state.get("round", 0), **details)

    def accept_tool_batch(messages, message, tools, role, state):
        """Validate the complete batch before adding history or executing tools."""
        allowed_names = [tool["function"]["name"] for tool in tools]
        limit = _batch_limit(message, role, allowed_names)
        remaining_tools = max(0, engine.max_calls - engine.calls)
        terminal_reserve = 1 if engine.assessed else 2
        affordable_citations = max(0, min(MAX_WRITER_CITATION_BATCH, remaining_tools - terminal_reserve))
        try:
            calls = _calls(message, limit=limit)
            if limit == MAX_WRITER_CITATION_BATCH and len(calls) > affordable_citations:
                raise ValueError("Citation batch exceeds the available global tool budget after reserving assessment and submission")
        except (ValueError, TypeError, KeyError) as exc:
            supplied = message.get("tool_calls") if isinstance(message, dict) else None
            count = len(supplied) if isinstance(supplied, list) else None
            event("agent_batch_rejected", role,
                  "The entire invalid tool batch was rejected before any tool execution",
                  state, actual_tool_call_count=count, maximum_tool_calls=limit,
                  remaining_tool_calls=remaining_tools,
                  maximum_affordable_citation_calls=affordable_citations if role == "writer" else None,
                  validation_error=str(exc)[:600])
            # Do not append an assistant message whose tool calls cannot be
            # answered. This repair consumes the existing bounded model turn.
            messages.append({"role": "user", "content": json.dumps({
                "error": str(exc)[:600], "actual_tool_call_count": count,
                "maximum_tool_calls": limit, "allowed_tool_names": allowed_names,
                "remaining_tool_calls": remaining_tools,
                "maximum_affordable_citation_calls": affordable_citations if role == "writer" else None,
                "reserved_terminal_tool_calls": terminal_reserve if role == "writer" else None,
                "instruction": (
                    "The entire previous tool batch was rejected; none of its calls executed. "
                    "Use only currently available tools and stay within the remaining global tool budget. "
                    + ("The writer may batch at most 24 calls only when EVERY call is cite_experiment_value or cite_web_excerpt; all mixed batches remain limited to 8. " if role == "writer" else "Batch at most 8 valid tool calls. ")
                    + "Do not assume any rejected call succeeded; leave tool calls for assessment and finish_report."
                ),
            })})
            return None
        messages.append(message)
        return calls

    def planner(state):
        round_number = state.get("round", 0) + 1
        state = {**state, "round": round_number}
        event(
            "agent_start",
            "planner",
            "Planner is assigning independent research questions",
            state,
        )
        catalogue = [
            {key: source.get(key) for key in ("id", "title", "kind", "uri", "metadata")}
            for source in list(engine.sources.values())
        ]
        messages = _messages(
            "planner",
            "Assign four separate TD3, SAC, federation-policy and literature goals using "
            "plan_research. Address reviewer gaps if this is a revision. State a short "
            "public rationale, not private reasoning. The literature_goal must describe only public "
            "scientific mechanisms: omit local source IDs, paths, artifact identifiers and private "
            "measurements. The three other specialists own local experiment evidence. "
            "Do not answer the research question.",
            {
                "question": state["question"],
                "round": round_number,
                "reviewer_gaps": state.get("gaps", []),
                "selected_evidence": catalogue,
            },
            SYSTEM,
        )
        for _ in range(3):
            message = engine.request(messages, [plan_tool])
            calls = accept_tool_batch(messages, message, [plan_tool], "planner", state)
            if calls is None:
                continue
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Call plan_research with td3_goal, sac_goal, federation_goal and literature_goal.",
                    }
                )
            for call in calls:
                try:
                    if call["function"]["name"] != "plan_research":
                        raise ValueError("The planner may only assign a research plan")
                    args = json.loads(call["function"]["arguments"])
                    if not isinstance(args, dict):
                        raise ValueError("Plan arguments must be an object")  # noqa: TRY004
                    plan = {
                        **{
                            name: _text(args.get(name), name)
                            for name in ROLE_GOALS.values()
                        },
                        "questions": _strings(
                            args.get("questions"), "questions", 8, 500
                        ),
                        "reason": _text(args.get("reason"), "plan reason"),
                    }
                    event(
                        "agent_tool",
                        "planner",
                        plan["reason"],
                        state,
                        tool="plan_research",
                    )
                    engine.execute(
                        "plan_research",
                        {"questions": plan["questions"], "reason": plan["reason"]},
                    )
                    engine.assessed = False
                    event(
                        "agent_complete",
                        "planner",
                        "Separate research assignments are ready",
                        state,
                    )
                    return {"round": round_number, "plan": plan, "catalogue": catalogue}
                except (ValueError, TypeError, KeyError) as exc:
                    _reply(messages, call, {"error": str(exc)[:1200]})
                    event("agent_tool_rejected", "planner", str(exc)[:1200], state)
        raise RuntimeError(
            "Planner could not produce a valid plan within its turn budget"
        )

    def researcher(state, role):
        local = role != "literature_researcher"
        allowed = (
            {"read_experiment", "inspect_federation"}
            if local
            else {"search_literature", "read_page"}
        )
        if role == "federation_strategist":
            allowed.update({"compare_clipping", "draft_policy_trial"})
        goal = state["plan"][ROLE_GOALS[role]]
        focus = {
            "td3_specialist": (
                "Audit heating TD3 action-coordinate scaling, action saturation and drift, "
                "available gradient/critic diagnostics, actor warm-up, and repair/adaptation costs. "
                "Distinguish observed telemetry from an untested mechanism. Actor/replay stay local. "
            ),
            "sac_specialist": (
                "Audit SAC deposition radius and power-fraction control, entropy-temperature "
                "learning and dependence on the TD3 microwave-power budget. Check coupled "
                "training and repair stability. Do not claim mirror steering or island suppression. "
            ),
            "federation_strategist": (
                "Audit separate heating_td3/deposition_sac channels and q0/q1 critic trunks, "
                "uniform versus clipped aggregation, source compatibility and missing donor deltas. "
                "Use compare_clipping for exact bounded mathematical diagnostics when a captured "
                "dossier supports it; these calculations do not predict policy performance. "
                "Use draft_policy_trial to obtain a typed next-run proposal and full cost accounting. "
                "Distinguish raw federation from accepted or rejected gate deployment; retain "
                "local, reset-only and initial-base controls. Never claim unperformed trials improved RL. "
                "Do not launch a shadow trial during parallel research. When shadow execution is "
                "enabled, a later decision stage will give you all four branches' findings and "
                "source snapshots before you decide whether a justified test is warranted. "
            ),
            "literature_researcher": (
                "Search public primary research through Tavily and extract relevant pages. "
                "Search general scientific concepts; do not transmit local result details. "
                "The TD3, SAC and federation specialists separately read all local records. "
                "You receive only a public literature assignment. Do not look for local records "
                "on the web or call read_page on local E/D source IDs. read_page accepts only "
                "web source IDs returned by search_literature. Your findings should explain "
                "published mechanisms and limits; the writer will connect them to local results. "
                "Prioritize two or three directly relevant primary papers, including counterevidence. "
                "That can be sufficient: assess relevance and submit findings rather than filling "
                "every remaining turn with loosely related papers. Expand only for a specific gap. "
            ),
        }[role]
        instructions = (
            focus
            + (
                "You are a LOCAL evidence specialist. The literature_researcher owns all web "
                "retrieval; do not search the web or invent a Tavily tool. Treat the shared system's "
                "search instructions as instructions to that separate role, not to you. "
                "Read each needed immutable record once using EITHER read_experiment OR "
                "inspect_federation: both return the same source content, so do not call both for "
                "one source. Inspect only the records needed for your assigned question. "
                "If a federation dossier or required telemetry is absent, use available summaries and "
                "explicitly list what is missing; do not invent diagnostics or require a tool to succeed. "
                if local
                else ""
            )
            + (
                "Your exact permitted tool names are: "
                + ", ".join([*sorted(allowed), "submit_findings"])
                + ". Use these names exactly; do not call a tool belonging to another role. "
                "Finish with submit_findings, including exact source IDs, a concise evidence-based "
                "summary and remaining gaps. Prefer a summary of about 1,200 characters, retaining "
                "the essential measured values, controls and limitations rather than repeating "
                "source content. Submit once the assigned question has been assessed. "
                "Empty evidence_ids is allowed only when no usable "
                "evidence was found, which you must state explicitly. A search snippet is not evidence. "
                "Do not issue a final answer or propose running commands."
            )
        )
        role_read_ids = set()
        supplied_sources = set()
        payload = {
            "question": state["question"],
            "assignment": goal,
            "reviewer_gaps": state.get("gaps", []),
            "selected_evidence": state["catalogue"],
        } if local else {
            "assignment": public_literature_text(goal, state["catalogue"]),
        }
        messages = _messages(
            role,
            instructions,
            payload,
            SYSTEM,
        )
        tools = [tools_by_name[name] for name in sorted(allowed)] + [findings_tool]
        event("agent_start", role, goal, state)
        for turn in range(6):
            # Stop gathering before the local limit (or shared budget) strands
            # read evidence without findings. Keep all final review calls free.
            remaining_calls = engine.max_turns - engine.model_calls
            if remaining_calls <= REVIEW_CALL_RESERVE:
                break
            terminal = turn == 5 or remaining_calls == REVIEW_CALL_RESERVE + 1
            turn_allowed = set() if terminal else allowed
            if terminal:
                messages.append({"role": "user", "content": json.dumps({
                    "instruction": "This is your final available turn. Call submit_findings now with the evidence you personally read, and explicitly state unresolved gaps. Do not search or read more sources.",
                    "eligible_evidence_ids": eligible_evidence_ids(role, role_read_ids),
                })})
                # Some compatible endpoints stall on a named forced tool even
                # when the same single-tool schema works with auto selection.
                # Execution still permits only submit_findings on this turn.
                tools_for_turn = [findings_tool]
            else:
                tools_for_turn = tools
            try:
                message = engine.request(messages, tools_for_turn, reserve_calls=REVIEW_CALL_RESERVE)
            except ModelCallBudgetReserved:
                break
            calls = accept_tool_batch(messages, message, tools_for_turn, role, state)
            if calls is None:
                continue
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Use your assigned tools, then submit_findings.",
                    }
                )
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                    if name == "submit_findings":
                        findings = _findings(args, engine, role, role_read_ids)
                        event(
                            "agent_complete",
                            role,
                            findings["summary"],
                            state,
                            evidence_ids=findings["evidence_ids"],
                            gaps=findings["gaps"],
                        )
                        return findings
                    if name not in turn_allowed:
                        raise ValueError("This tool is outside this researcher's role")
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")  # noqa: TRY004
                    if name == "test_shadow" and args.get("source_id") not in role_read_ids:
                        raise ValueError("The federation strategist must personally read its cited source before testing a shadow change")
                    event(
                        "agent_tool",
                        role,
                        _text(args.get("reason"), "tool reason"),
                        state,
                        tool=name,
                    )
                    result = engine.execute(name, args)
                    if isinstance(result, dict):
                        returned_source = result.get("source", result)
                        if (
                            isinstance(returned_source, dict)
                            and (
                                returned_source.get("id") in engine.read_ids
                                or (role == "literature_researcher" and name == "read_page"
                                    and returned_source.get("kind") == "web")
                            )
                        ):
                            role_read_ids.add(returned_source["id"])
                except (ValueError, TypeError, KeyError) as exc:
                    result = {
                        "error": str(exc)[:1200],
                        "allowed_tool_names": [*sorted(turn_allowed), "submit_findings"],
                    }
                    if name == "submit_findings":
                        result["eligible_evidence_ids"] = eligible_evidence_ids(role, role_read_ids)
                    event(
                        "agent_tool_rejected", role, result["error"], state, tool=name
                    )
                _reply(messages, call, result, supplied_sources)
            if terminal:
                break
        retained_ids = eligible_evidence_ids(role, role_read_ids)
        event(
            "agent_complete",
            role,
            "Researcher reached its turn budget with unresolved gaps; personally read evidence is retained",
            state,
            evidence_ids=retained_ids,
        )
        return {
            "summary": "Incomplete specialist assessment: no accepted findings were submitted before this role's turn budget. Personally read source snapshots are retained for direct writer and judge inspection; no specialist conclusion is certified.",
            "evidence_ids": retained_ids,
            "gaps": [f"{role} did not complete its evidence assessment. Retained sources are read evidence, not reviewed specialist conclusions."],
        }

    def td3_specialist(state):
        return {"td3_findings": researcher(state, "td3_specialist")}

    def sac_specialist(state):
        return {"sac_findings": researcher(state, "sac_specialist")}

    def federation_strategist(state):
        return {"federation_findings": researcher(state, "federation_strategist")}

    def literature_researcher(state):
        return {"literature_findings": researcher(state, "literature_researcher")}

    def shadow_decision(state):
        """Reuse the federation role after all research, never before it."""
        role = "federation_strategist"
        previous = state["federation_findings"]
        if engine.shadow_attempted:
            retained = [source["id"] for source in list(engine.sources.values())
                        if source.get("metadata", {}).get("adapter") == "shadow_execution"]
            event(
                "shadow_decision_skipped", role,
                "The single shadow attempt has already been used; review its retained evidence",
                state,
            )
            return {"federation_findings": {
                **previous,
                "evidence_ids": list(dict.fromkeys([*previous["evidence_ids"], *retained])),
            }}
        # Leave room for all six writer turns and one judge call. A lack of
        # budget is a documented skip, never a silent forced experiment.
        if engine.max_turns - engine.model_calls <= REVIEW_CALL_RESERVE:
            event("shadow_decision_skipped", role,
                  "The remaining model-call budget is reserved for the writer and judge", state)
            return {"federation_findings": {
                **previous,
                "gaps": [*previous["gaps"], "No shadow decision was made because the remaining model-call budget was reserved for review."],
            }}
        selected = {
            identifier
            for key in ROLE_FINDINGS.values()
            for identifier in state[key]["evidence_ids"]
        }
        sources = [source for source in list(engine.sources.values()) if source["id"] in selected]
        supplied_sources = {(source["id"], source["sha256"]) for source in sources}
        messages = _messages(
            role,
            "Post-research shadow decision: inspect all four completed branches and their actual "
            "source snapshots, including literature and counterevidence. Decide whether one allowed "
            "bounded change has a scientific justification. You may skip: submit_findings with your "
            "reason and gaps if a test is unsupported, redundant or unlikely to answer the question. "
            "There is no required candidate or desired positive result. If testing, first personally "
            "call read_experiment for the supporting local source in this stage, then test_shadow. "
            "Your exact permitted tool names are read_experiment, test_shadow and submit_findings. "
            "Do not search the web; the literature branch has already supplied extracted sources. "
            "Read each needed local record at most once in this stage, even if a tool is rejected. "
            "If a read acknowledges a content_reference, its full immutable source already appears "
            "earlier in this context; use that exact snapshot without asking for another copy. "
            "The pilot retains local/reset-only/uniform/clipped controls, but lacks joint SAC and "
            "initial-base arms. Inspect measured outcomes or failures before submit_findings. "
            "Only local evidence IDs you read in this stage or a returned shadow source belong in "
            "your new findings; original specialist/literature evidence is retained separately. "
            "Prefer about 1,200 characters of new findings with essential measured comparisons and gaps. "
            "Do not imply one seed establishes improvement. Preserve negative outcomes and uncertainty.",
            {
                "phase": "post_research_shadow_decision",
                "question": state["question"],
                "reviewer_gaps": state.get("gaps", []),
                **{key: state[key] for key in ROLE_FINDINGS.values()},
                "sources": source_views(sources, reference_texts=[state[key]["summary"] for key in ROLE_FINDINGS.values()]),
            },
            SYSTEM,
        )
        allowed = {"read_experiment", "test_shadow"}
        tools = [tools_by_name[name] for name in sorted(allowed)] + [findings_tool]
        role_read_ids = set()
        event("agent_start", role,
              "Federation strategist is deciding whether to test after reading all specialist findings",
              state, phase="post_research_shadow_decision")
        for _ in range(6):
            if engine.max_turns - engine.model_calls <= REVIEW_CALL_RESERVE:
                break
            try:
                message = engine.request(messages, tools, reserve_calls=REVIEW_CALL_RESERVE)
            except ModelCallBudgetReserved:
                break
            calls = accept_tool_batch(messages, message, tools, role, state)
            if calls is None:
                continue
            if not calls:
                messages.append({"role": "user", "content": "Use the bounded tools or submit_findings explaining why no test is warranted."})
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                    if name == "submit_findings":
                        findings = _findings(args, engine, role, role_read_ids)
                        event("agent_complete", role, findings["summary"], state,
                              phase="post_research_shadow_decision",
                              evidence_ids=findings["evidence_ids"], gaps=findings["gaps"])
                        return {"federation_findings": {
                            "summary": previous["summary"] + "\nPost-research shadow decision: " + findings["summary"],
                            "evidence_ids": list(dict.fromkeys([*previous["evidence_ids"], *findings["evidence_ids"]])),
                            "gaps": list(dict.fromkeys([*previous["gaps"], *findings["gaps"]])),
                        }}
                    if name not in allowed:
                        raise ValueError("This tool is outside the post-research shadow decision")
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")  # noqa: TRY004
                    if name == "test_shadow" and args.get("source_id") not in role_read_ids:
                        raise ValueError("The federation strategist must personally read its cited source in the decision stage before testing a shadow change")
                    event("agent_tool", role, _text(args.get("reason"), "tool reason"),
                          state, tool=name, phase="post_research_shadow_decision")
                    result = engine.execute(name, args)
                    if isinstance(result, dict) and result.get("id") in engine.read_ids:
                        role_read_ids.add(result["id"])
                except (ValueError, TypeError, KeyError) as exc:
                    result = {
                        "error": str(exc)[:1200],
                        "allowed_tool_names": [*sorted(allowed), "submit_findings"],
                    }
                    if name == "submit_findings":
                        result["eligible_evidence_ids"] = eligible_evidence_ids(role, role_read_ids)
                    event("agent_tool_rejected", role, result["error"], state,
                          tool=name, phase="post_research_shadow_decision")
                _reply(messages, call, result, supplied_sources)
        # Even if the model fails to submit findings, retain any measured trial
        # or failure source so the writer and judge can inspect it.
        shadow_ids = [identifier for identifier in role_read_ids
                      if engine.sources[identifier].get("metadata", {}).get("adapter") == "shadow_execution"]
        return {"federation_findings": {
            **previous,
            "evidence_ids": list(dict.fromkeys([*previous["evidence_ids"], *shadow_ids])),
            "gaps": [*previous["gaps"], "The post-research shadow decision reached its bounded model-call limit without submitting findings."],
        }}

    def writer(state, *, repair=False):
        event(
            "agent_start",
            "writer",
            "Writer is repairing the judged draft using captured evidence" if repair else "Writer is comparing four independently collected specialist findings",
            state,
        )
        selected = {
            identifier
            for key in ROLE_FINDINGS.values()
            for identifier in state[key]["evidence_ids"]
        }
        if repair:
            selected.update(engine.read_ids)
        sources = [
            source for source in engine.sources.values() if source["id"] in selected
        ]
        payload = {
            "question": state["question"],
            **{key: state[key] for key in ROLE_FINDINGS.values()},
            "sources": source_views(sources, reference_texts=[state[key]["summary"] for key in ROLE_FINDINGS.values()]),
        }
        reflection_instructions = ""
        protected = {}
        if repair:
            protected = _protected_claims(state["report"])
            payload.update(phase="writer_reflection", reviewed_draft=copy.deepcopy(state["report"]),
                           judge_gaps=state.get("gaps", []), remaining_writer_calls=3,
                           protected_claims=copy.deepcopy(list(protected.values())))
            reflection_instructions = (
                "This is the single bounded reflection pass on the actual reviewed draft. Read its "
                "per-claim verdicts, citation coverage failures, proposal judgments and judge gaps. "
                "The server owns protected_claims, which already passed traceability and semantic review. "
                "Do not rewrite their text, IDs, quotes or offsets: the server retains them unchanged "
                "even if omitted or resubmitted differently. Revise flagged claims, proposals, limitations "
                "and stop_reason only. New claims still need exact evidence. The fresh judge will recheck "
                "ALL claims, including protected ones, and may reject any of them. "
                "Repair with already captured sources: obtain needed exact citations in your first "
                "turn, reassess evidence, then submit the whole corrected report. Narrow or omit "
                "uncovered clauses instead of inventing support; acceptance is not required. "
                "Configuration values and settings are substantive claims and need their own citations. "
                "Use one precise outcome label: a /deployed citation does not establish /raw behavior, "
                "even if values happen to match elsewhere. Reconcile limitations and stop_reason with "
                "later recorded tool outcomes; an earlier unavailable helper does not imply all shadow "
                "execution paths were unavailable. Keep raw learner and gate-deployed endpoints distinct "
                "in proposals and report both where relevant. Derive proposed budgets from the captured "
                "configuration, counting configurations and seeds, and do not add evaluation subsets twice. "
                "Label gate and initial-evaluation counts as subsets of total evaluation costs when recorded "
                "that way. Proposed additional development seeds are not known to be unused unless the seed "
                "inventory was checked. Evidence absent from a selected snapshot is not necessarily absent "
                "from the project. If claims are withheld, stop_reason must not say all measurements were cited. "
                "Prefer the actual primary paper to a secondary or AI-generated summary. State cross-seed "
                "behavior as a hypothesis to test, not a guaranteed prediction. "
                "No new searches, source reads or physics runs are allowed. "
                "There are at most three writer turns and then a fresh independent judge. "
            )
            engine.assessed = False
        messages = _messages(
            "writer",
            reflection_instructions + (
                "Synthesize all four specialist branches, checking their actual source snapshots. "
                "Specialist prose is unverified: compare its numerical assertions against the source, "
                "including the direction of inequalities, and discard unsupported interpretations. "
                "Experiment citation pointers are relative to the parsed JSON inside source.content: "
                "do not prepend /content, the source ID, a filename or an archival raw-result pointer. "
                "Use field names actually present in that snapshot, never a guessed alias. Copy an "
                "exact returned JSON value at that pointer as the quote. Prefer the smallest complete "
                "recorded object containing the claimed subject and values when its quotation fits "
                "2000 characters. One such citation can cover several clauses; use scalar citations "
                "for additional settings absent from that object. Call cite_experiment_value for the "
                "chosen object, then copy its exact returned quotation; never reconstruct a subset. "
                "Batch lookups when useful. A writer batch may contain up to 24 calls ONLY if "
                "every call is cite_experiment_value or cite_web_excerpt; mixed batches including "
                "assessment or finish_report remain limited to 8. The global tool budget still applies: "
                "consult tool_budget, refreshed each turn, and stay within maximum_citation_calls_this_turn; "
                "prioritize required citations and leave calls for assessment and submission. "
                "Give path as an array of string dictionary keys "
                "and integer array indices, e.g. [\"seeds\", \"17\", \"cost\"] or "
                "[\"by_seed\", 0, \"mean_cost_difference\"]. Copy each returned source_id, locator "
                "and quote unchanged into claim evidence. This helper resolves existing records "
                "only; it does not establish causality or truth. If a value is too large, follow "
                "its citation_path_hint to select a smaller complete child object or scalar. "
                "For a web source call cite_web_excerpt with one "
                "short exact anchor phrase and max_words <=25, then copy source_id, locator "
                "and quote unchanged. If repeated anchors return candidates, choose one using "
                "its context and copy only its citation object unchanged. Use one short contiguous quote per web source; never join "
                "different passages with ellipses. cite_experiment_value is only for local JSON. "
                "Use no more than 25 unique quoted words total from each web source across the report. "
                "Use recorded computed outputs for aggregation diagnostics and typed trial proposals. "
                "Distinguish mathematical bounds from measured control performance; never promise "
                "RL improvement from an unrun policy. Explicitly report missing telemetry or incompatible schemas. "
                "Call assess_evidence, then finish_report with direct arguments answer, claims, "
                "next_runs, limitations and stop_reason. Keep claims and next_runs as native arrays "
                "of objects. Do not wrap these fields in report or report_json, and do not serialize "
                "arrays or objects as JSON strings. Set answer to the empty string; the server "
                "reconstructs it from checked claims, so do not generate duplicate answer prose. "
                "If evidence is insufficient, state the gaps instead of filling them with plausible facts. "
                "An incomplete specialist branch may still supply read source snapshots. Inspect those "
                "snapshots directly; do not claim retrieval found nothing merely because its findings "
                "were incomplete, and do not treat retained sources as reviewed specialist conclusions. "
                "Long web snapshots may be explicit partial verbatim excerpts with canonical hashes "
                "and offsets. Cite only supplied passages; omissions do not prove absence. "
                "Keep next runs as bounded proposals only, retaining fair controls and budgets. "
                "Your tools cannot gather additional evidence."
            ),
            payload,
            SYSTEM,
        )
        allowed = {"assess_evidence", "cite_experiment_value", "cite_web_excerpt", "finish_report"}
        writer_turns = 3 if repair else 6
        for turn in range(writer_turns):
            remaining_tools = max(0, engine.max_calls - engine.calls)
            terminal_reserve = 1 if engine.assessed else 2
            payload["tool_budget"] = {
                "maximum_total_calls": engine.max_calls, "used_calls": engine.calls,
                "remaining_calls": remaining_tools, "reserved_terminal_calls": terminal_reserve,
                "maximum_citation_calls_this_turn": max(0, min(MAX_WRITER_CITATION_BATCH, remaining_tools - terminal_reserve)) if turn < writer_turns - 2 else 0,
                "maximum_mixed_batch_calls": MAX_TOOL_BATCH,
            }
            # Refresh the same context field each turn without burying the latest
            # tool results or rejection feedback under another user message.
            messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
            turn_allowed = allowed
            if turn >= writer_turns - 2:
                turn_allowed = {"assess_evidence"} if turn == writer_turns - 2 and not engine.assessed else {"finish_report"}
                messages.append({"role": "user", "content": (
                    "Citation lookup is now closed. Use already resolved citations, omit unresolved "
                    "claims, and list their gaps explicitly. Do not invent evidence or call more lookup tools. "
                    "The 24-call citation allowance no longer applies; submit only the requested terminal tool. "
                    + ("Call assess_evidence now, marking enough false if required support is unresolved. "
                       "Your next and final turn is reserved for finish_report."
                       if "assess_evidence" in turn_allowed else
                       "Call finish_report now using only supported observations and explicit limitations. "
                       "An insufficient report is valid; do not claim acceptance or fill evidence gaps.")
                )})
                event("writer_terminal_turn", "writer", "Reserving the remaining writer turns for assessment and a structured draft", state,
                      turn=turn + 1, allowed_tools=sorted(turn_allowed))
            tools = [tools_by_name[name] for name in sorted(turn_allowed)]
            message = engine.request(messages, tools)
            calls = accept_tool_batch(messages, message, tools, "writer", state)
            if calls is None:
                continue
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Assess the evidence, then submit your structured report with finish_report.",
                    }
                )
            for call in calls:
                name = call["function"]["name"]
                try:
                    if name not in turn_allowed:
                        raise ValueError(
                            "This tool is unavailable on the current writer turn; allowed tools: "
                            + ", ".join(sorted(turn_allowed))
                        )
                    args = json.loads(call["function"]["arguments"])
                    if repair and name == "finish_report":
                        args, preserved = _preserve_protected_claims(args, protected)
                        event("writer_protected_claims_preserved", "writer",
                              "Previously supported and traceable claims were preserved unchanged; all merged claims still require fresh review",
                              state, **preserved, fresh_review_required=True)
                    event(
                        "agent_tool",
                        "writer",
                        "Writer requested " + name,
                        state,
                        tool=name,
                    )
                    result = engine.execute(name, args)
                    if name == "finish_report" and "report" in result:
                        event(
                            "agent_complete",
                            "writer",
                            "Structured draft is ready for independent judgment",
                            state,
                        )
                        return {"draft": result["report"]}
                except (ValueError, TypeError, KeyError) as exc:
                    result = {"error": str(exc)[:1200]}
                    citation_failures = getattr(exc, "citation_failures", None)
                    if citation_failures:
                        result["citation_failures"] = citation_failures
                    citation_path_hint = getattr(exc, "citation_path_hint", None)
                    if citation_path_hint:
                        result["citation_path_hint"] = citation_path_hint
                    event(
                        "agent_tool_rejected",
                        "writer",
                        result["error"],
                        state,
                        tool=name,
                    )
                _reply(messages, call, result)
        raise RuntimeError(
            "Writer could not produce a valid draft within its turn budget"
        )

    def writer_reflection(state):
        count = state.get("writer_reflections", 0) + 1
        event("writer_reflection", "writer", "Revisiting the judged draft without new retrieval or experiments",
              state, reflection=count, maximum_writer_calls=3, reserved_judge_calls=1)
        try:
            result = writer(state, repair=True)
        except RuntimeError as exc:
            retained = copy.deepcopy(state["report"])
            retained["evidence_sufficient"] = False
            retained["orchestration"].update(writer_reflections=count, status="insufficient_evidence")
            retained["limitations"].append("The bounded writer reflection did not produce a valid replacement; the prior independently judged report is retained as insufficient evidence.")
            event("writer_reflection_failed", "writer", str(exc)[:1200], state, reflection=count)
            return {"report": retained, "writer_reflections": count, "reflection_draft_ready": False}
        return {**result, "writer_reflections": count, "reflection_draft_ready": True}

    def repairable_review(report):
        return any(
            claim.get("kind") != "hypothesis" and (
                claim.get("verification", {}).get("status") == "unsupported"
                or claim.get("review", {}).get("verdict") in {"uncertain", "unsupported"}
            ) for claim in report["claims"]
        ) or any(verdict.get("verdict") != "adequate" for verdict in report.get("judge", {}).get("proposal_verdicts", []))

    def judge(state):
        event(
            "agent_start",
            "judge",
            "Independent judge is testing claim support and proposal quality",
            state,
        )
        report = engine.review(copy.deepcopy(state["draft"]), state["question"])
        judgment = report.get("judge", {})
        decision = judgment.get("decision", "insufficient")
        gaps = judgment.get("gaps", [])
        if decision not in {"accept", "revise", "insufficient"}:
            decision = "insufficient"
        if not isinstance(gaps, list) or any(not isinstance(gap, str) for gap in gaps):
            gaps = ["The judge did not provide valid evidence gaps."]
        gaps = [gap[:600] for gap in gaps[:12]]
        report["judge"] = {**judgment, "decision": decision, "gaps": gaps}
        accepted = decision == "accept" and report.get("evidence_sufficient") is True
        if not accepted:
            report["evidence_sufficient"] = False
        report["orchestration"] = {
            "framework": "LangGraph",
            "agents": list(AGENTS),
            "rounds": state["round"],
            "max_rounds": MAX_ROUNDS,
            "status": "accepted" if accepted else "insufficient_evidence",
            "parallel_researchers": list(RESEARCHERS),
            "writer_reflections": state.get("writer_reflections", 0),
            "max_writer_reflections": MAX_WRITER_REFLECTIONS,
        }
        if decision == "revise" and state["round"] >= MAX_ROUNDS:
            report["limitations"].append(
                "The judge requested further work, but the two-round research limit was reached."
            )
        remaining_calls = max(0, engine.max_turns - engine.model_calls)
        if not accepted and repairable_review(report) and state.get("writer_reflections", 0) < MAX_WRITER_REFLECTIONS and remaining_calls < 4:
            report["limitations"].append("Fewer than four model calls remain for one writer reflection and independent recheck; the judged report remains insufficient.")
            event("writer_reflection_deferred", "judge", "Insufficient remaining calls for bounded writer reflection and independent recheck",
                  state, remaining_model_calls=remaining_calls, minimum_reflection_calls=4)
        if decision == "revise" and state["round"] < MAX_ROUNDS and remaining_calls < 12:
            report["orchestration"]["revision_deferred"] = "insufficient_model_call_budget"
            report["orchestration"]["remaining_model_calls"] = remaining_calls
            report["limitations"].append(
                f"The judge requested revision, but only {remaining_calls} model calls remain; "
                "fewer than the 12-call minimum reserved for another research round. "
                "The independently judged report is retained as insufficient evidence."
            )
            event(
                "agent_revision_deferred", "judge",
                "Revision deferred because the remaining model-call budget cannot cover another research round",
                state, remaining_model_calls=remaining_calls, minimum_revision_calls=12,
                decision=decision, evidence_sufficient=False,
            )
        event(
            "agent_complete",
            "judge",
            "Judge decision: " + decision,
            state,
            decision=decision,
            gaps=gaps,
            evidence_sufficient=accepted,
        )
        return {"report": report, "gaps": gaps}

    def after_judge(state):
        if (
            not state["report"].get("evidence_sufficient")
            and repairable_review(state["report"])
            and state.get("writer_reflections", 0) < MAX_WRITER_REFLECTIONS
            and engine.max_turns - engine.model_calls >= 4
        ):
            return "writer_reflection"
        if (
            state["report"]["judge"]["decision"] == "revise"
            and state["round"] < MAX_ROUNDS
            and not state["report"]["orchestration"].get("revision_deferred")
        ):
            event(
                "agent_revision",
                "judge",
                "Reviewer gaps are returning to the planner",
                state,
                gaps=state["gaps"],
            )
            return "planner"
        return END

    graph = StateGraph(ResearchState)
    for name, node in (
        ("planner", planner),
        ("td3_specialist", td3_specialist),
        ("sac_specialist", sac_specialist),
        ("federation_strategist", federation_strategist),
        ("literature_researcher", literature_researcher),
        ("writer", writer),
        ("judge", judge),
    ):

        def scoped(state, *, _node=node, _name=name):
            with engine.agent(_name):
                return _node(state)

        graph.add_node(name, scoped)
    graph.add_edge(START, "planner")
    for role in RESEARCHERS:
        graph.add_edge("planner", role)
    if getattr(engine, "shadow_executor", None) is not None:
        def scoped_shadow(state):
            with engine.agent("federation_strategist"):
                return shadow_decision(state)

        graph.add_node("shadow_decision", scoped_shadow)
        graph.add_edge(list(RESEARCHERS), "shadow_decision")
        graph.add_edge("shadow_decision", "writer")
    else:
        graph.add_edge(list(RESEARCHERS), "writer")
    graph.add_edge("writer", "judge")
    def scoped_reflection(state):
        with engine.agent("writer"):
            return writer_reflection(state)

    graph.add_node("writer_reflection", scoped_reflection)
    graph.add_conditional_edges("writer_reflection", lambda state: "judge" if state.get("reflection_draft_ready") else END,
                                {"judge": "judge", END: END})
    graph.add_conditional_edges("judge", after_judge, {"writer_reflection": "writer_reflection", "planner": "planner", END: END})
    return graph.compile()


def run_research_graph(engine, question):
    """Run a bounded live multi-agent graph and return its final reviewed report."""
    graph = build_research_graph(engine)
    from langsmith import tracing_context

    # The user authorized Nebius/Tavily, not an implicit third-party trace sink.
    with tracing_context(enabled=False):
        result = graph.invoke(
            {
                "question": _text(question, "research question", 4000),
                "round": 0,
                "gaps": [],
            },
            config={"recursion_limit": 16, "max_concurrency": 4},
        )
    report = result["report"]
    report["usage"] = {
        "model_calls": engine.model_calls,
        "tokens": engine.total_usage,
        "searches": engine.searches,
        "extractions": engine.extracts,
        "tool_calls": engine.calls,
    }
    return report
