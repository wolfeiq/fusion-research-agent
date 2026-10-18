"""Web quotation helpers preserve contiguous stored text and citation limits."""

import json
from copy import deepcopy

import pytest

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.verification import validate_report
from fusion_agent.research.web_citations import cite_web_excerpt


def source(content="Clipping can limit large updates. Separate paragraph.", **overrides):
    return {
        "id": "W001", "kind": "web", "uri": "https://example.org/paper",
        "metadata": {"status": "extracted"}, "content": content, **overrides,
    }


def check_quote(saved, citation):
    _, start, end = citation["locator"].split(":")
    assert saved["content"][int(start):int(end)] == citation["quote"]
    return validate_report({
        "answer": "", "claims": [{
            "id": "C1", "text": "A scoped statement requiring semantic review.",
            "kind": "literature", "confidence": "medium", "evidence": [citation],
        }], "next_runs": [], "limitations": [], "stop_reason": "Transport check.",
    }, [saved])


def test_exact_offsets_and_multiline_whitespace_are_preserved_without_mutation():
    saved = source("Preface.\nClipping \tcan\nlimit large  updates.\nMore text.")
    before = deepcopy(saved)
    citation = cite_web_excerpt(saved, "Clipping can limit", 5)
    assert citation["quote"] == "Clipping \tcan\nlimit large  updates."
    assert saved == before
    report = check_quote(saved, citation)
    assert report["claims"][0]["verification"]["status"] == "traceable"
    assert report["verification"]["semantic_review_required"]


def test_helper_does_not_join_two_legitimate_passages_with_invented_ellipsis():
    saved = source("First valid passage. Middle words. Second valid passage.")
    with pytest.raises(ValueError, match="not found"):
        cite_web_excerpt(saved, "First valid passage ... Second valid passage", 12)


@pytest.mark.parametrize("content,anchor", [
    ("unclipping may differ", "clipping"), ("clipping may differ", "clip"),
])
def test_anchor_cannot_start_or_end_inside_a_whitespace_delimited_word(content, anchor):
    with pytest.raises(ValueError, match="not found"):
        cite_web_excerpt(source(content), anchor, 3)


@pytest.mark.parametrize("content,anchor", [
    ("repeat text then repeat text", "repeat text"), ("a a a", "a a"),
])
def test_ambiguous_including_overlapping_anchors_return_exact_candidates(content, anchor):
    saved = source(content)
    result = cite_web_excerpt(saved, anchor, 5)
    assert result["total_matches"] == 2 and result["omitted_matches"] == 0
    assert len(result["candidates"]) == 2
    for candidate in result["candidates"]:
        assert len(candidate["context_before"]) <= 160
        assert len(candidate["context_after"]) <= 160
        assert check_quote(saved, candidate["citation"])["verification"]["traceable_claims"] == 1


def test_repeated_anchor_candidates_are_bounded_and_not_silently_selected():
    saved = source(" ".join(["Repeated phrase here."] * 20))
    result = cite_web_excerpt(saved, "Repeated phrase", 3)
    assert result["total_matches"] == 20 and result["omitted_matches"] == 15
    assert len(result["candidates"]) == 5 and "quote" not in result
    assert len({item["citation"]["locator"] for item in result["candidates"]}) == 5


def test_engine_reports_ambiguous_candidates_without_logging_source_text():
    events = []
    engine = ResearchEngine([source("Repeated phrase. Middle. Repeated phrase.")], None, None, events.append)
    engine.planned = True
    engine.read_ids.add("W001")
    result = engine.execute("cite_web_excerpt", {
        "source_id": "W001", "anchor": "Repeated phrase.", "max_words": 2, "reason": "Resolve occurrences.",
    })
    assert len(result["candidates"]) == 2
    assert events[-1]["candidate_count"] == 2
    assert "quote" not in events[-1] and "context_before" not in events[-1]


@pytest.mark.parametrize("anchor", ["missing phrase", "clipping can", "Clipping ... limit"])
def test_anchor_matching_changes_only_whitespace_not_case_or_punctuation(anchor):
    with pytest.raises(ValueError, match="not found"):
        cite_web_excerpt(source(), anchor, 10)


@pytest.mark.parametrize("budget", [0, 26, True, 1.5, "5"])
def test_word_budget_is_strictly_bounded_integer(budget):
    with pytest.raises(ValueError, match="integer from 1 to 25"):
        cite_web_excerpt(source(), "Clipping", budget)


def test_whole_anchor_must_fit_budget_and_quotes_cannot_exceed_character_limit():
    with pytest.raises(ValueError, match="entire anchor"):
        cite_web_excerpt(source(), "Clipping can limit", 2)
    with pytest.raises(ValueError, match="2000 characters"):
        cite_web_excerpt(source("begin " + "x" * 2000), "begin", 2)
    with pytest.raises(ValueError, match="300 characters"):
        cite_web_excerpt(source(), "x" * 301, 25)
    with pytest.raises(ValueError, match="control"):
        cite_web_excerpt(source(), "bad\x00anchor", 25)


def test_quote_uses_at_most_the_available_words_up_to_the_requested_limit():
    saved = source("Only three words.")
    assert cite_web_excerpt(saved, "Only", 25)["quote"] == saved["content"]


@pytest.mark.parametrize("overrides", [
    {"kind": "search"}, {"kind": "experiment"}, {"kind": "protocol"},
    {"metadata": {"status": "discovery_only"}}, {"uri": "http://127.0.0.1/private"},
    {"uri": "file:///private"},
])
def test_nonextracted_or_nonpublic_sources_are_rejected(overrides):
    with pytest.raises(ValueError):
        cite_web_excerpt(source(**overrides), "Clipping", 5)


def test_engine_requires_registered_read_web_source_and_exact_envelope():
    saved = source()
    events = []
    engine = ResearchEngine([saved], None, None, events.append)
    engine.planned = True
    args = {"source_id": "W001", "anchor": "Clipping", "max_words": 5, "reason": "Copy one exact passage."}
    with pytest.raises(ValueError, match="Read an extracted"):
        engine.execute("cite_web_excerpt", args)
    engine.read_ids.add("W001")
    with pytest.raises(ValueError, match="exactly"):
        engine.execute("cite_web_excerpt", {**args, "url": "https://other.org"})
    citation = engine.execute("cite_web_excerpt", args)
    assert citation["quote"] == "Clipping can limit large updates."
    assert events[-1]["kind"] == "cite_web_excerpt"
    assert "quote" not in events[-1]
    assert check_quote(saved, citation)["verification"]["traceable_claims"] == 1


def test_real_read_page_path_registers_extracted_source_as_read_without_new_helper_fetch():
    class Retrieval:
        calls = 0

        def extract(self, url):
            self.calls += 1
            return {"results": [{"url": url, "raw_content": "Clipping can limit large updates."}]}

    saved = source(kind="search", metadata={"status": "discovery_only"})
    retrieval = Retrieval()
    engine = ResearchEngine([saved], None, retrieval, lambda _event: None)
    engine.planned = True
    engine.execute("read_page", {"source_id": "W001", "reason": "Read the source."})
    citation = engine.execute("cite_web_excerpt", {
        "source_id": "W001", "anchor": "Clipping can", "max_words": 5, "reason": "Resolve exact text.",
    })
    assert "W001" in engine.read_ids
    assert retrieval.calls == 1
    assert citation["quote"] == "Clipping can limit large updates."


def test_two_separate_excerpts_do_not_bypass_total_web_word_limit():
    saved = source(" ".join(f"word{number}" for number in range(50)))
    first = cite_web_excerpt(saved, "word0 word1", 20)
    second = cite_web_excerpt(saved, "word30 word31", 20)
    report = {
        "answer": "", "claims": [{
            "id": "C1", "text": "Claim with too many quoted words.",
            "kind": "literature", "confidence": "medium", "evidence": [first, second],
        }], "next_runs": [], "limitations": [], "stop_reason": "Budget check.",
    }
    result = validate_report(report, [saved])
    assert result["claims"][0]["verification"]["status"] == "unsupported"
    assert "25-word" in json.dumps(result)


def test_actual_pdf_math_controls_fail_at_helper_without_rewriting_the_snapshot():
    # The saved W005 PDF in attempt 19 returned exactly this unusable 20-word
    # span at chars:7291:7367. The anchor is clean; the chosen following math is not.
    excerpt = "clip(gt, c) = gt · min \x1a 1, c ∥gt∥ \x1b .\n(3) However, DP is more complex in FL"
    saved = source(" " * 7291 + excerpt)
    original = deepcopy(saved)
    assert saved["content"][7291:7367] == excerpt
    with pytest.raises(ValueError, match=r"U\+001A, U\+001B.*clean prose anchor"):
        cite_web_excerpt(saved, "clip(gt, c)", 20)
    assert saved == original
    # Choosing clean prose changes only the selected span, never the source.
    citation = cite_web_excerpt(saved, "However, DP", 8)
    assert citation["quote"] == "However, DP is more complex in FL"
    assert check_quote(saved, citation)["verification"]["traceable_claims"] == 1
    assert saved == original


@pytest.mark.parametrize("codepoint", [code for code in range(32) if chr(code) not in "\n\r\t"])
def test_every_forbidden_report_string_control_is_rejected_even_after_a_clean_anchor(codepoint):
    saved = source("Clean anchor " + chr(codepoint) + " following prose.")
    original = deepcopy(saved)
    with pytest.raises(ValueError, match="unsupported control characters.*clean prose anchor"):
        cite_web_excerpt(saved, "Clean anchor", 5)
    assert saved == original


@pytest.mark.parametrize("whitespace", ["\n", "\r", "\t", "\r\n", "\u00a0"])
def test_supported_whitespace_citations_keep_exact_offsets_and_pass_report_validation(whitespace):
    saved = source("Clean anchor" + whitespace + "following prose.")
    citation = cite_web_excerpt(saved, "Clean anchor", 4)
    assert citation["quote"] == saved["content"]
    assert check_quote(saved, citation)["verification"]["traceable_claims"] == 1


def test_repeated_anchor_filters_invalid_candidates_and_accounts_for_each_match():
    saved = source("Repeated anchor \x1a math. Repeated anchor clean prose. Repeated anchor \x1b math.")
    original = deepcopy(saved)
    result = cite_web_excerpt(saved, "Repeated anchor", 4)
    assert result["total_matches"] == 3
    assert result["omitted_matches"] == 0
    assert len(result["rejected_candidates"]) == 2
    assert [item["codepoints"] for item in result["rejected_candidates"]] == [["U+001A"], ["U+001B"]]
    assert len(result["candidates"]) == 1
    for candidate in result["candidates"]:
        citation = candidate["citation"]
        assert citation["quote"] == "Repeated anchor clean prose."
        assert check_quote(saved, citation)["verification"]["traceable_claims"] == 1
    assert "rejected" in result["instruction"]
    assert saved == original


def test_rejected_repeated_matches_are_separate_from_unexamined_matches():
    saved = source("Repeated anchor \x1a math. " + "Repeated anchor clean prose. " * 7)
    result = cite_web_excerpt(saved, "Repeated anchor", 4)
    assert result["total_matches"] == 8
    assert len(result["candidates"]) == 4
    assert len(result["rejected_candidates"]) == 1
    assert result["omitted_matches"] == 3
    for candidate in result["candidates"]:
        assert check_quote(saved, candidate["citation"])["verification"]["traceable_claims"] == 1


def test_all_examined_repeated_occurrences_with_controls_fail_with_actionable_feedback():
    saved = source("Repeated anchor \x1a math. Repeated anchor \x1b math.")
    with pytest.raises(ValueError, match=r"U\+001A, U\+001B.*clean prose anchor"):
        cite_web_excerpt(saved, "Repeated anchor", 4)
