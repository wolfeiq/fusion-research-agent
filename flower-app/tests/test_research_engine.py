"""Injected-model research-loop and provider-boundary tests; no network calls."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.providers import (
    NebiusModel,
    ProviderError,
    Settings,
    Tavily,
    public_url,
)

URL = "https://arxiv.org/abs/1802.09477"
PAGE = "Results should be evaluated across independent training seeds."


def local_source(identifier="E001"):
    content = json.dumps({"status": "complete_verified", "metrics": {"cost": 0.12}})
    return {
        "id": identifier, "title": "Saved development audit", "kind": "experiment",
        "uri": "results/td3_heating_stability/seed17/independent_analysis.json",
        "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "metadata": {"status": "complete_verified"},
    }


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"call_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }]}


def report(*, overclaim=False):
    claims = [
        {"id": "C001", "text": "The saved development endpoint cost is 0.12.",
         "kind": "observation", "confidence": "high", "evidence": [
             {"source_id": "E001", "quote": "0.12", "locator": "/metrics/cost"}]},
        {"id": "C002", "text": "The extracted paper recommends evaluating independent training seeds.",
         "kind": "literature", "confidence": "medium", "evidence": [
             {"source_id": "W002", "quote": PAGE, "locator": ""}]},
    ]
    if overclaim:
        claims.append({
            "id": "C003", "text": "Federation is proven to work on all real reactors.",
            "kind": "observation", "confidence": "high", "evidence": [
                {"source_id": "E001", "quote": "0.12", "locator": "/metrics/cost"}],
        })
    return {
        "answer": "UNCITED MODEL SUMMARY MUST BE REMOVED",
        "claims": claims,
        "next_runs": [{
            "title": "Paired development comparison",
            "hypothesis": "Compatible donors may help a scarce-data target.",
            "changes": "Freeze normalized critic inputs and three development seeds.",
            "controls": ["local", "reset-only", "uniform", "clipped"],
            "metrics": ["Raw endpoint cost", "Checkpoint competence", "Donor shots"],
            "budget": "Three seeds and 24 target shots per arm; source costs separate.",
            "stop_rule": "Stop when the frozen budget is consumed or physics fails.",
            "evidence_ids": ["E001", "W002"],
            "limitations": "Simulation and development evidence, not hardware safety.",
        }],
        "limitations": ["One saved development seed is not a confirmation study."],
        "stop_reason": "Enough evidence for this scoped answer; general benefit remains unresolved.",
    }


def verdicts(*, overclaim=False, omit=False):
    items = [{"id": "C001", "verdict": "supported", "reason": "The saved scalar matches."}]
    if not omit:
        items.append({"id": "C002", "verdict": "supported", "reason": "The extracted sentence supports this wording."})
    if overclaim:
        items.append({"id": "C003", "verdict": "unsupported", "reason": "A simulated endpoint cannot establish universal hardware transfer."})
    for item in items:
        supported = item["verdict"] == "supported"
        item.update(all_clauses_supported=supported, scope_matches=supported,
                    unsupported_clauses=[] if supported else [item["reason"]])
    return call("review_claims", verdicts=items)


def research_script(data=None, review=None):
    return [
        call("plan_research", questions=["What does the saved experiment show?", "What counts as adequate seed evidence?"], reason="Separate measured results from claims about generalization."),
        call("read_experiment", source_id="E001", reason="Read the raw-versus-development scope."),
        call("search_literature", query="reinforcement learning independent seeds evaluation methodology", reason="Find primary methodological evidence."),
        call("read_page", source_id="W002", reason="Read the primary paper before citing it."),
        call("assess_evidence", enough=True, gaps=["No held-out federation confirmation"], reason="The data support a bounded conclusion and a next test."),
        call("finish_report", report_json=json.dumps(data or report())),
        review or verdicts(),
    ]


class FakeModel:
    def __init__(self, script):
        self.script = deepcopy(script)
        self.requests = []

    def complete(self, messages, tools, tool_choice="auto"):
        self.requests.append(deepcopy((messages, tools, tool_choice)))
        assert self.script, "Unexpected model request"
        return self.script.pop(0), {
            "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "cached": True,
        }


class FakeRetrieval:
    def __init__(self, *, extracted_url=URL, page=PAGE):
        self.extracted_url, self.page = extracted_url, page
        self.searches, self.extracts = [], []

    def search(self, query):
        self.searches.append(query)
        return {"results": [{"url": URL, "title": "Primary paper", "content": PAGE}],
                "usage": {"credits": 2}}

    def extract(self, url):
        self.extracts.append(url)
        return {"results": [{"url": self.extracted_url, "raw_content": self.page}],
                "usage": {"credits": 1}}


def make_engine(script=(), **kwargs):
    events = []
    model, retrieval = FakeModel(script), FakeRetrieval()
    engine = ResearchEngine([local_source()], model, retrieval, events.append, **kwargs)
    return engine, model, retrieval, events


def prepare(engine, *, extract=True):
    engine.execute("plan_research", {"questions": ["What is supported?"], "reason": "Plan before retrieval."})
    engine.execute("read_experiment", {"source_id": "E001", "reason": "Read the saved evidence."})
    engine.execute("search_literature", {"query": "TD3 evaluation methodology", "reason": "Find relevant primary research."})
    if extract:
        engine.execute("read_page", {"source_id": "W002", "reason": "Read primary content."})
    engine.execute("assess_evidence", {"enough": True, "gaps": [], "reason": "Enough for a scoped report."})


def drive_tools_and_review(engine, question):
    """Exercise model/tool boundaries; graph routing has separate tests."""
    from fusion_agent.research.engine import TOOLS

    messages = [{"role": "user", "content": question}]
    for _ in range(6):
        message = engine.request(messages, TOOLS)
        messages.append(message)
        tool_call = message["tool_calls"][0]
        result = engine.execute(
            tool_call["function"]["name"],
            json.loads(tool_call["function"]["arguments"]),
        )
        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result)})
        if "report" in result:
            return engine.review(result["report"], question)
    raise AssertionError("The fixture did not finish a report")


def test_complete_model_directed_run_preserves_sources_trace_and_usage():
    engine, model, retrieval, events = make_engine(research_script())
    result = drive_tools_and_review(engine, "Does this evidence establish a cold-start advantage?")
    assert not model.script
    assert retrieval.extracts == [URL]
    assert len(retrieval.searches) == 1
    assert engine.sources["W002"]["kind"] == "web"
    assert engine.sources["W002"]["sha256"] == hashlib.sha256(PAGE.encode()).hexdigest()
    assert engine.sources["E001"] == local_source()
    assert engine.model_calls == 7
    assert engine.total_usage == {"prompt_tokens": 70, "completion_tokens": 28, "total_tokens": 98}
    assert (engine.searches, engine.extracts, engine.calls) == (1, 1, 6)
    assert "UNCITED MODEL" not in result["answer"]
    assert "[E001]" in result["answer"] and "[W002]" in result["answer"]
    assert result["next_runs"][0]["status"] == "proposal_only"
    assert [e["kind"] for e in events if e["kind"] in {
        "plan", "read_experiment", "search", "extract", "assessment", "review", "review_complete",
    }] == ["plan", "read_experiment", "search", "extract", "assessment", "review", "review_complete"]
    assert [e["usage"] for e in events if e["kind"] == "retrieval_usage"] == [{"credits": 2}, {"credits": 1}]
    review_messages, _, choice = model.requests[-1]
    assert len(review_messages) == 2  # Fresh reviewer context, not the planner transcript.
    assert choice["function"]["name"] == "review_claims"
    reviewed = json.loads(review_messages[1]["content"])
    assert {s["id"] for s in reviewed["sources"]} == {"E001", "W002"}


def test_semantic_reviewer_downgrades_locatable_but_unsupported_overclaim():
    engine, _, _, _ = make_engine(research_script(report(overclaim=True), verdicts(overclaim=True)))
    result = drive_tools_and_review(engine, "Is federation established?")
    claim = result["claims"][2]
    assert claim["verification"]["status"] == "unsupported"
    assert claim["confidence"] == "low"
    assert "all real reactors" not in result["answer"]
    assert result["verification"]["traceable_claims"] == 2
    assert result["verification"]["unsupported_claims"] == 1


def test_missing_reviewer_verdict_is_uncertain_and_removed_from_answer():
    engine, _, _, _ = make_engine(research_script(review=verdicts(omit=True)))
    result = drive_tools_and_review(engine, "Assess the evidence")
    assert result["claims"][1]["review"]["verdict"] == "uncertain"
    assert result["claims"][1]["verification"]["status"] == "unsupported"
    assert "extracted paper recommends" not in result["answer"]


GOOD_RUBRIC = {
    "citation_support": 2,
    "experimental_fairness": 2,
    "actionability": 2,
    "scope_honesty": 2,
}


def accepting_review(rubric=GOOD_RUBRIC):
    response = verdicts()
    args = json.loads(response["tool_calls"][0]["function"]["arguments"])
    args.update(
        decision="accept",
        gaps=[],
        proposal_verdicts=[{
            "index": 0, "verdict": "adequate",
            "reason": "The bounded paired comparison includes costs and stopping rules.",
        }],
        rubric=rubric,
    )
    response["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    return response


def test_proposal_only_source_identity_does_not_expose_uncited_content_to_claim_judge():
    engine, model, _, _ = make_engine([accepting_review()])
    prepare(engine)
    extra = local_source("E002")
    extra["content"] = '{"source_training_episodes":48}'
    engine.sources["E002"] = extra
    engine.execute("read_experiment", {"source_id": "E002", "reason": "Account for donor work in the proposal."})
    draft = report()
    draft["next_runs"][0]["evidence_ids"].append("E002")
    validated = engine.execute("finish_report", {"report_json": json.dumps(draft)})["report"]
    engine.review(validated, "Which experiment should we run next?")
    payload = json.loads(model.requests[0][0][1]["content"])
    assert {item["id"] for item in payload["sources"]} == {"E001", "W002", "E002"}
    assert len(payload["sources"]) == 3
    assert all("content" not in item for item in payload["sources"])
    assert "source_training_episodes" not in json.dumps(payload)
    assert all(e["source_id"] != "E002" for c in payload["claims"] for e in c["evidence"])


def test_judge_accepts_supported_report_with_all_four_good_scores():
    engine, model, _, _ = make_engine(research_script(review=accepting_review()))
    result = drive_tools_and_review(engine, "Assess this bounded proposal")
    assert result["evidence_sufficient"] is True
    assert result["judge"]["decision"] == "accept"
    assert result["judge"]["rubric"] == GOOD_RUBRIC
    assert result["judge"]["gaps"] == []
    assert "0 = unmet" in model.requests[-1][0][0]["content"]
    assert "1 = partially met" in result["judge"]["rubric_note"]


@pytest.mark.parametrize("rubric", [
    {},
    {key: value for key, value in GOOD_RUBRIC.items() if key != "actionability"},
    {**GOOD_RUBRIC, "citation_support": 0},
    {**GOOD_RUBRIC, "scope_honesty": True},
    {**GOOD_RUBRIC, "experimental_fairness": "2"},
    {**GOOD_RUBRIC, "actionability": 3},
    {**GOOD_RUBRIC, "actionability": -1},
    [],
    None,
])
def test_judge_cannot_accept_missing_invalid_or_unmet_rubric(rubric):
    engine, _, _, _ = make_engine(research_script(review=accepting_review(rubric)))
    result = drive_tools_and_review(engine, "Assess this bounded proposal")
    assert result["judge"]["decision"] == "revise"
    assert any("all four rubric dimensions" in gap for gap in result["judge"]["gaps"])


def test_partial_rubric_score_can_accompany_acceptance_when_nothing_is_unmet():
    rubric = {**GOOD_RUBRIC, "actionability": 1}
    engine, _, _, _ = make_engine(research_script(review=accepting_review(rubric)))
    result = drive_tools_and_review(engine, "Assess this bounded proposal")
    assert result["judge"]["decision"] == "accept"
    assert result["judge"]["rubric"]["actionability"] == 1


def test_bad_report_schema_is_rejected_before_review():
    engine, model, _, _ = make_engine()
    prepare(engine)
    data = report()
    data["launch_command"] = "do not execute"
    with pytest.raises(ValueError):
        engine.execute("finish_report", {"report_json": json.dumps(data)})
    assert not model.requests


def test_selected_but_unread_local_source_cannot_be_cited():
    engine, _, _, _ = make_engine()
    prepare(engine)
    engine.sources["E002"] = local_source("E002")
    data = report()
    data["claims"][0]["evidence"][0]["source_id"] = "E002"
    with pytest.raises(ValueError, match="Read each local source"):
        engine.execute("finish_report", {"report_json": json.dumps(data)})


def test_reviewer_cannot_upgrade_a_wrong_quote_to_traceable():
    data = report(overclaim=True)
    data["claims"][2]["evidence"][0]["quote"] = "NONEXISTENT QUOTE"
    review = verdicts(overclaim=True)
    judgments = json.loads(review["tool_calls"][0]["function"]["arguments"])
    judgments["verdicts"][2]["verdict"] = "supported"
    judgments["verdicts"][2].update(all_clauses_supported=True, scope_matches=True, unsupported_clauses=[])
    review["tool_calls"][0]["function"]["arguments"] = json.dumps(judgments)
    engine, _, _, _ = make_engine(research_script(data, review))
    result = drive_tools_and_review(engine, "Assess the evidence")
    assert result["claims"][2]["review"]["verdict"] == "supported"
    assert result["claims"][2]["verification"]["status"] == "unsupported"
    assert "all real reactors" not in result["answer"]


@pytest.mark.parametrize("updates", [
    {"all_clauses_supported": False}, {"scope_matches": False},
    {"unsupported_clauses": ["One numerical clause lacks its own citation."]},
    {"all_clauses_supported": 1}, {"scope_matches": "true"},
    {"unsupported_clauses": "[]"}, {"unsupported_clauses": ["gap"] * 13},
    {"unsupported_clauses": [""]},
])
def test_structured_coverage_conflict_cannot_be_hidden_behind_supported_enum(updates):
    response = accepting_review()
    args = json.loads(response["tool_calls"][0]["function"]["arguments"])
    args["verdicts"][0].update(updates)
    response["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    engine, _, _, _ = make_engine(research_script(review=response))
    result = drive_tools_and_review(engine, "Check full citation coverage.")
    reviewed_claim = result["claims"][0]
    assert reviewed_claim["review"]["raw_verdict"] == "supported"
    assert reviewed_claim["review"]["verdict"] == "uncertain"
    assert reviewed_claim["verification"]["status"] == "unsupported"
    assert reviewed_claim["text"] not in result["answer"]
    assert result["judge"]["decision"] == "revise"
    assert result["evidence_sufficient"] is False


def test_missing_structured_coverage_is_not_accepted_as_legacy_success():
    response = accepting_review()
    args = json.loads(response["tool_calls"][0]["function"]["arguments"])
    del args["verdicts"][0]["scope_matches"]
    response["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    engine, _, _, _ = make_engine(research_script(review=response))
    result = drive_tools_and_review(engine, "Require explicit scope judgment.")
    assert result["claims"][0]["review"]["coverage_fields_valid"] is False
    assert result["claims"][0]["review"]["verdict"] == "uncertain"


@pytest.mark.parametrize("decision", ["revise", "insufficient"])
def test_standalone_review_marks_evidence_insufficient_when_final_decision_is_not_accept(decision):
    response = accepting_review()
    args = json.loads(response["tool_calls"][0]["function"]["arguments"])
    args["decision"] = decision
    response["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    engine, _, _, _ = make_engine(research_script(review=response))
    result = drive_tools_and_review(engine, "Check final sufficiency independently of graph wrapper.")
    assert all(claim["verification"]["status"] == "traceable" for claim in result["claims"])
    assert result["judge"]["decision"] == decision
    assert result["evidence_sufficient"] is False


def test_finish_requires_an_explicit_evidence_assessment():
    engine, _, _, _ = make_engine()
    engine.execute("plan_research", {"questions": ["Evidence?"], "reason": "Plan."})
    engine.execute("read_experiment", {"source_id": "E001", "reason": "Read."})
    with pytest.raises(ValueError, match="assess evidence"):
        engine.execute("finish_report", {"report_json": json.dumps(report())})


@pytest.mark.parametrize("name,args", [
    ("search_literature", {"query": "TD3", "reason": "Search."}),
    ("read_experiment", {"source_id": "E001", "reason": "Read."}),
    ("finish_report", {"report_json": json.dumps(report())}),
])
def test_retrieval_and_finish_require_a_plan(name, args):
    engine, model, retrieval, _ = make_engine()
    with pytest.raises(ValueError, match="[Pp]lan"):
        engine.execute(name, args)
    assert not retrieval.searches and not retrieval.extracts and not model.requests


def test_wrong_local_quote_cannot_be_promoted_by_finish():
    engine, _, _, _ = make_engine()
    prepare(engine)
    data = report()
    data["claims"][0]["evidence"][0]["quote"] = "0.13"
    with pytest.raises(ValueError, match="traceable experiment"):
        engine.execute("finish_report", {"report_json": json.dumps(data)})


def test_discovery_snippet_cannot_substitute_for_page_extraction():
    engine, _, retrieval, _ = make_engine()
    prepare(engine, extract=False)
    with pytest.raises(ValueError, match="extracted, cited literature"):
        engine.execute("finish_report", {"report_json": json.dumps(report())})
    assert not retrieval.extracts


@pytest.mark.parametrize("source_id", ["W999", "https://example.org/unregistered", "file:///private/key"])
def test_unregistered_url_extraction_is_rejected(source_id):
    engine, _, retrieval, _ = make_engine()
    engine.execute("plan_research", {"questions": ["What is supported?"], "reason": "Plan."})
    with pytest.raises(ValueError, match="discovered"):
        engine.execute("read_page", {"source_id": source_id, "reason": "Read page."})
    assert not retrieval.extracts


@pytest.mark.parametrize("query", [
    "Find flower-app/.dashboard/shadow/43d55b69d02440b3a019cf9b64f6845a/comparison.json",
    "Find 43d55b69d02440b3a019cf9b64f6845a clipping diagnostics",
    "Search trial 43d55b69-d024-40b3-a019-cf9b64f6845a",
    r"Search C:\Users\researcher\experiments\summary.json",
    "file:///private/experiments/summary.json",
    "file%3A%2F%2F%2Fprivate%2Fsummary.json",
    r"Search \\server\research\summary.json",
    "Search /home/researcher/experiments/summary.json",
    "Search ../private/summary.json",
    "Search results/td3_heating_stability/seed17/independent_analysis.json",
    "Find local source E001",
])
def test_private_artifact_query_is_rejected_before_tavily(query):
    engine, _, retrieval, events = make_engine()
    engine.execute("plan_research", {"questions": ["What is supported?"], "reason": "Plan."})
    with pytest.raises(ValueError, match="Public search query rejected"):
        engine.execute("search_literature", {"query": query, "reason": "Find evidence."})
    assert retrieval.searches == [] and engine.searches == 0
    assert not any(event["kind"] == "search" for event in events)


@pytest.mark.parametrize("query", [
    "TD3/SAC federated reinforcement learning clipped critic updates",
    "arXiv:1802.09477 Addressing Function Approximation Error in Actor-Critic Methods",
    "https://arxiv.org/abs/1802.09477",
    "arXiv hep-th/9901001",
    "DOI 10.1109/TNNLS.2018.2866075",
    "https://doi.org/10.1038/s41586-021-04301-9",
])
def test_public_scientific_query_doi_and_arxiv_remain_allowed(query):
    engine, _, retrieval, _ = make_engine()
    engine.execute("plan_research", {"questions": ["What is supported?"], "reason": "Plan."})
    engine.execute("search_literature", {"query": query, "reason": "Find primary evidence."})
    assert retrieval.searches == [query]


def test_read_page_local_source_redirects_to_literature_completion():
    engine, _, retrieval, _ = make_engine()
    engine.execute("plan_research", {"questions": ["What is supported?"], "reason": "Plan."})
    with pytest.raises(ValueError, match="submit_findings") as error:
        engine.execute("read_page", {"source_id": "E001", "reason": "Read this source."})
    assert "local" in str(error.value).lower() and "specialists" in str(error.value)
    assert retrieval.extracts == []


def test_report_citation_failure_preserves_exact_claim_pointer_and_actionable_keys():
    engine, _, _, events = make_engine()
    prepare(engine)
    draft = report()
    draft["claims"][0]["evidence"][0]["locator"] = "/clipping_diagnostics/not_a_real_key"
    with pytest.raises(ValueError, match="traceable experiment") as caught:
        engine.execute("finish_report", {"report": draft})
    feedback = caught.value.citation_failures
    failed = feedback["claims"][0]
    assert failed["id"] == "C001"
    assert failed["evidence"] == [{
        "source_id": "E001", "locator": "/clipping_diagnostics/not_a_real_key",
        "reason": "The experiment JSON pointer does not resolve.",
    }]
    assert feedback["local_source_top_level_keys"]["E001"] == {
        "keys": ["status", "metrics"], "omitted_count": 0,
    }
    saved = next(event for event in events if event["kind"] == "report_citation_validation")
    assert saved["rejected_claims"][0]["text"] == draft["claims"][0]["text"]
    assert saved["rejected_claims"][0]["evidence"][0]["quote"] == "0.12"
    assert saved["rejected_claims"][0]["verification"]["status"] == "unsupported"
    draft["claims"][0]["evidence"][0]["locator"] = "/metrics/cost"
    fixed = engine.execute("finish_report", {"report": draft})
    assert fixed["report"]["claims"][0]["verification"]["status"] == "traceable"


def test_report_citation_feedback_is_bounded_and_web_overflow_quotes_are_not_saved():
    engine, _, _, events = make_engine()
    prepare(engine)
    draft = report()
    draft["claims"][0]["evidence"] *= 8
    for evidence in draft["claims"][0]["evidence"]:
        evidence["locator"] = "/missing"
    draft["claims"].append(deepcopy(draft["claims"][0]))
    draft["claims"][-1]["id"] = "C003"
    draft["claims"][1]["evidence"][0]["quote"] = "over limit " * 14
    with pytest.raises(ValueError) as caught:
        engine.execute("finish_report", {"report": draft})
    feedback = caught.value.citation_failures
    assert sum(len(claim["evidence"]) for claim in feedback["claims"]) == 8
    assert feedback["omitted_evidence_count"] == 9
    saved = next(event for event in events if event["kind"] == "report_citation_validation")
    web_failure = next(claim for claim in saved["rejected_claims"] if claim["id"] == "C002")
    assert web_failure["evidence"][0]["quote"] == ""


def test_extracted_content_cannot_be_attributed_to_a_different_url():
    engine, _, retrieval, _ = make_engine()
    prepare(engine, extract=False)
    retrieval.extracted_url = "https://example.org/unrelated"
    with pytest.raises(ValueError):
        engine.execute("read_page", {"source_id": "W002", "reason": "Read paper."})
    assert engine.sources["W002"]["kind"] == "search"


def test_large_extraction_records_explicit_excerpt_and_both_hashes():
    engine, _, retrieval, _ = make_engine()
    prepare(engine, extract=False)
    retrieval.page = "a" * 25_000
    source = engine.execute("read_page", {"source_id": "W002", "reason": "Read paper."})
    assert len(source["content"]) == 24_000
    assert source["metadata"]["truncated"] is True
    assert source["sha256"] == hashlib.sha256(source["content"].encode()).hexdigest()
    assert source["metadata"]["full_extraction_sha256"] == hashlib.sha256(retrieval.page.encode()).hexdigest()


def test_call_budget_exhaustion_prevents_further_tools_or_report():
    engine, _, retrieval, _ = make_engine(max_calls=1)
    engine.execute("plan_research", {"questions": ["Evidence?"], "reason": "Plan."})
    with pytest.raises(RuntimeError, match="tool-call budget"):
        engine.execute("search_literature", {"query": "TD3", "reason": "Search."})
    assert not retrieval.searches


def test_time_budget_exhaustion_does_not_call_provider():
    engine, model, _, _ = make_engine(max_seconds=-1)
    with pytest.raises(RuntimeError, match="time budget"):
        engine.request([], [])
    assert not model.requests


def test_settings_keep_secrets_out_of_public_state_and_repr(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "NEBIUS_SECRET_MARKER")
    monkeypatch.setenv("TAVILY_API_KEY", "TAVILY_SECRET_MARKER")
    settings = Settings.from_env()
    settings.validate()
    public = json.dumps(settings.public()) + repr(settings)
    assert "SECRET_MARKER" not in public
    assert settings.public()["nebius"] is True
    assert settings.public()["tavily"] is True


@pytest.mark.parametrize("endpoint", [
    "http://api.example.org/v1", "https://api.example.org/v1?key=SECRET",
    "https://api.example.org/v1#SECRET", "https://user:SECRET@api.example.org/v1",
    "https://127.0.0.1/v1", "https://localhost/v1", "https://api.example.org:8443/v1",
])
@pytest.mark.parametrize("field", ["base_url", "tavily_url"])
def test_provider_endpoint_validation_requires_public_https(endpoint, field):
    settings = Settings(nebius_key="n", tavily_key="t", **{field: endpoint})
    with pytest.raises(ValueError):
        settings.validate()
    assert "SECRET" not in json.dumps(settings.public())


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://169.254.169.254/latest", "https://[::1]/", "https://host.internal/", "https://host.local/", "https://user:key@example.org/"])
def test_private_and_credentialed_web_targets_rejected(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_nebius_errors_do_not_reveal_provider_body_or_key():
    class UpstreamError(Exception):
        status_code = 401

    def fail(**_kwargs):
        raise UpstreamError("NEBIUS_SECRET_MARKER provider response body")

    model = object.__new__(NebiusModel)
    model.model = "test-model"
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(ProviderError) as raised:
        model.complete([], [])
    assert "HTTP 401" in str(raised.value)
    assert "SECRET" not in str(raised.value) and raised.value.__suppress_context__


def test_nebius_retains_tool_calls_and_usage_but_not_private_reasoning():
    message = SimpleNamespace(
        content=None, reasoning_content="PRIVATE REASONING", reasoning="PRIVATE REASONING",
        tool_calls=[SimpleNamespace(id="call1", function=SimpleNamespace(name="plan_research", arguments='{"reason":"Plan"}'))],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7))
    captured = {}

    def respond(**kwargs):
        captured.update(kwargs)
        return response

    model = object.__new__(NebiusModel)
    model.model = "test-model"
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
    result, usage = model.complete([], [])
    assert "PRIVATE" not in json.dumps(result)
    assert result["tool_calls"][0]["function"]["name"] == "plan_research"
    assert usage["total_tokens"] == 7 and captured["model"] == "test-model"


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload
        self.read_limits = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        self.read_limits.append(limit)
        return self.payload[:limit]


def patch_http(monkeypatch, payload=b'{}', error=None):
    import fusion_agent.research.providers as module

    response = FakeHTTPResponse(payload)
    captured = {}

    def open_request(req, timeout):
        captured.update(request=req, timeout=timeout)
        if error:
            raise error
        return response

    def opener(handler):
        captured["handler"] = handler
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(module, "build_opener", opener)
    return response, captured


def test_tavily_uses_bounded_advanced_requests_without_following_redirects(monkeypatch):
    response, captured = patch_http(monkeypatch, b'{"results":[],"usage":{"credits":2}}')
    client = Tavily(Settings(tavily_key="SECRET_TOKEN"))
    result = client.search("TD3 benchmark methodology")
    assert result["usage"] == {"credits": 2}
    req = captured["request"]
    assert req.full_url == "https://api.tavily.com/search"
    assert json.loads(req.data)["search_depth"] == "advanced"
    assert json.loads(req.data)["include_raw_content"] is False
    assert "SECRET_TOKEN" not in req.full_url and "SECRET_TOKEN" not in req.data.decode()
    assert captured["handler"].redirect_request(None, None, None, None, None, None) is None
    assert response.read_limits == [2_000_001]
    client.extract(URL)
    assert json.loads(captured["request"].data)["urls"] == [URL]


@pytest.mark.parametrize("payload", [b'[]', b'not json', b'{"invalid":'])
def test_tavily_invalid_json_becomes_safe_error(monkeypatch, payload):
    patch_http(monkeypatch, payload)
    with pytest.raises(ProviderError, match="invalid JSON"):
        Tavily(Settings(tavily_key="secret")).search("TD3")


def test_tavily_rejects_oversize_response_without_unbounded_read(monkeypatch):
    response, _ = patch_http(monkeypatch, b'x' * 2_100_000)
    with pytest.raises(ProviderError, match="retrieval limit"):
        Tavily(Settings(tavily_key="secret")).search("TD3")
    assert response.read_limits == [2_000_001]


@pytest.mark.parametrize("error", [
    HTTPError("https://api.tavily.com/search?secret=TOKEN", 401, "BODY SECRET", {}, None),
    URLError("TAVILY_SECRET_MARKER"),
])
def test_tavily_error_redacts_upstream_details(monkeypatch, error):
    patch_http(monkeypatch, error=error)
    with pytest.raises(ProviderError) as raised:
        Tavily(Settings(tavily_key="secret")).search("TD3")
    assert "SECRET" not in str(raised.value) and "TOKEN" not in str(raised.value)


def test_tavily_private_extraction_rejected_before_http(monkeypatch):
    _, captured = patch_http(monkeypatch)
    with pytest.raises(ValueError):
        Tavily(Settings(tavily_key="secret")).extract("http://127.0.0.1/key")
    assert "request" not in captured
