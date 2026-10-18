"""Source projection preserves exact evidence without oversized handoffs."""
from __future__ import annotations

import copy
import hashlib
import json

from fusion_agent.research.source_views import source_views


def web(identifier="W001", text=None):
    content = text or ("Introductory context. " * 600 +
                       "A qualification before. The claimed benefit requires compatible tasks. A counterpoint after. " +
                       "Additional material. " * 600)
    return {"id": identifier, "kind": "web", "title": "Synthetic source fixture",
            "uri": "https://example.org/paper", "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(), "metadata": {}}


def test_views_preserve_exact_spans_surrounding_citations_hashes_and_original_sources():
    source = web()
    original = copy.deepcopy(source)
    quote = "The claimed benefit requires compatible tasks."
    view = source_views([source], quotes_by_source={"W001": [quote]})[0]
    projected = view["content"]
    assert source == original
    assert projected["representation"] == "partial_verbatim_excerpts"
    assert projected["canonical_sha256"] == source["sha256"] == view["sha256"]
    assert projected["included_characters"] <= 8000
    assert projected["omitted_characters"] > 0
    assert projected["canonical_characters"] == len(source["content"])
    assert projected["matched_quotes_not_included"] == 0
    for excerpt in projected["excerpts"]:
        assert excerpt["text"] == source["content"][excerpt["start"]:excerpt["end"]]
    cited = next(excerpt["text"] for excerpt in projected["excerpts"] if quote in excerpt["text"])
    assert "A qualification before." in cited and "A counterpoint after." in cited
    assert "omitted text may contain qualifications or counterevidence" in projected["notice"]


def test_handoff_quotes_are_matched_with_whitespace_but_never_invented():
    source = web(text="Beginning. " * 1000 + "A measured\n result has limits." + "Ending. " * 2000)
    view = source_views([source], reference_texts=['The paper says "A measured result has limits." and "an invented finding".'])[0]
    excerpts = view["content"]["excerpts"]
    assert any("A measured\n result has limits." in excerpt["text"] for excerpt in excerpts)
    assert not any("an invented finding" in excerpt["text"] for excerpt in excerpts)


def test_numeric_records_and_short_pages_stay_complete_and_copied():
    local = {"id": "E001", "kind": "experiment", "content": json.dumps({"cost": 0.123, "nested": [1, 2, 3]}), "sha256": "fixture"}
    short = web(text="Short complete paper.")
    originals = copy.deepcopy([local, short])
    views = source_views([local, short])
    assert views == originals
    views[0]["content"] = "changed view"
    assert [local, short] == originals


def test_realistic_six_paper_handoff_plus_new_pilot_fits_original_context_guard():
    # Reproduces the source sizes of the failed live handoff, without depending
    # on private/generated files in .dashboard. Add a new 22k pilot and report.
    sources = [web(f"W{i:03d}", ("Paper context with Unicode αβγ. " * 1000)[:24000]) for i in range(6)]
    for i, size in enumerate((22301, 29769, 2561, 3792, 3788, 3788, 22301)):
        content = json.dumps({f"cost_{j}": j / 1000 for j in range(size // 25)})
        content += " " * max(0, size - len(content))
        sources.append({"id": f"E{i:03d}", "kind": "experiment", "content": content,
                        "sha256": hashlib.sha256(content.encode()).hexdigest()})
    canonical = copy.deepcopy(sources)
    payload = {"sources": source_views(sources), "findings": "Scoped findings. " * 1000,
               "claims": "Report and tool-call overhead. " * 400}
    messages = [{"role": "system", "content": "Instructions. " * 500},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    assert len(json.dumps(messages)) < 260000
    assert sources == canonical


def test_review_receives_quote_centered_views_but_keeps_canonical_citation_sources():
    from fusion_agent.research.engine import ResearchEngine
    from fusion_agent.research.verification import validate_report

    source = web()
    quote = "The claimed benefit requires compatible tasks."

    class Reviewer:
        payload = None

        def complete(self, messages, tools, tool_choice="auto"):
            self.payload = json.loads(messages[1]["content"])
            args = {"verdicts": [{"id": "C1", "verdict": "supported", "reason": "The scoped claim matches the cited passage."}],
                    "decision": "insufficient", "gaps": ["No measured experiment claim."],
                    "proposal_verdicts": [], "rubric": {"citation_support": 2, "experimental_fairness": 1,
                                                        "actionability": 1, "scope_honesty": 2}}
            return {"tool_calls": [{"function": {"name": "review_claims", "arguments": json.dumps(args)}}]}, {}

    reviewer = Reviewer()
    engine = ResearchEngine([source], reviewer, None, lambda event: None)
    report = validate_report({"answer": "", "claims": [{"id": "C1", "text": quote, "kind": "literature", "confidence": "medium",
                              "evidence": [{"source_id": "W001", "quote": quote, "locator": ""}]}],
                              "next_runs": [], "limitations": [], "stop_reason": "Fixture inspection."}, [source])
    result = engine.review(report, "What does this primary source support?")
    projected = reviewer.payload["claims"][0]["evidence"][0]
    assert any(quote in excerpt["text"] for excerpt in projected["excerpts"])
    assert engine.sources["W001"] == source and isinstance(source["content"], str)
    assert result["claims"][0]["verification"]["status"] == "traceable"
    assert any("bounded verbatim excerpts" in limitation for limitation in result["limitations"])
