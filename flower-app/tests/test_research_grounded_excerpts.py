"""Exact quotation catalogue checks; no provider calls or scientific assertions."""

import hashlib
from copy import deepcopy

import pytest

from fusion_agent.research.grounded_excerpts import (
    MAX_PER_SOURCE,
    MAX_TOTAL,
    build_excerpt_catalogue,
)
from fusion_agent.research.web_citations import cite_web_excerpt


def source(content, identifier="W001", uri="https://proceedings.mlr.press/v162/zhang22b.html", **updates):
    result = {"id": identifier, "kind": "web", "uri": uri, "title": "Captured primary source",
              "content": content, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
              "metadata": {"status": "extracted"}}
    result.update(updates)
    return result


def assert_exact(entry, captured):
    prefix, start, end = entry["locator"].split(":")
    start, end = int(start), int(end)
    assert prefix == "chars" and captured["content"][start:end] == entry["quote"]
    assert entry["context"]["before"] == captured["content"][max(0, start - 220):start]
    assert entry["context"]["after"] == captured["content"][end:end + 220]
    assert 1 <= len(entry["quote"].split()) <= 25
    assert entry["source_sha256"] == captured["sha256"]
    request = entry["request"]
    actual = cite_web_excerpt(captured, request["anchor"], request["max_words"])
    assert actual == {"source_id": captured["id"], "quote": entry["quote"], "locator": entry["locator"]}


def test_complete_six_word_sentence_is_exact_and_hash_bound():
    captured = source("Clipping limits transmitted updates before aggregation.")
    original = deepcopy(captured)
    entries = build_excerpt_catalogue(captured, "Does clipping limit updates?")
    assert len(entries) == 1
    entry = entries[0]
    assert entry["quote"] == captured["content"]
    assert entry["selection_kind"] == "complete_sentence"
    assert_exact(entry, captured)
    assert captured == original


def test_unicode_whitespace_and_offsets_are_preserved_without_normalization():
    captured = source("Abstract\n\nHeating αβ control preserves naïve plasma profiles.\n\nThe μ-wave policy\tchooses a stable\n deposition position.")
    entries = build_excerpt_catalogue([captured], "plasma profiles")
    assert len(entries) == 2
    assert any("\t" in entry["quote"] and "\n" in entry["quote"] for entry in entries)
    assert any("naïve" in entry["quote"] for entry in entries)
    for entry in entries:
        assert_exact(entry, captured)


def test_long_sentence_uses_literal_bounded_spans_not_generated_ellipsis():
    text = ("The controllers coordinate their heating actions while the experiment records observed temperature profiles "
            "and shared critic updates across several independent development seeds before comparing each candidate "
            "against its frozen baseline under the same recorded conditions to identify limitations and document uncertainty.")
    captured = source(text)
    entries = build_excerpt_catalogue(captured, "critic updates")
    assert entries and all(entry["selection_kind"] == "bounded_span" for entry in entries)
    for entry in entries:
        assert "…" not in entry["quote"]
        assert_exact(entry, captured)


def test_shared_short_anchor_extends_until_unique_without_guessing_offsets():
    captured = source("The recorded evaluation uses several independent development seeds. The recorded evaluation uses several independent confirmation seeds.")
    entries = build_excerpt_catalogue(captured)
    assert len(entries) == 2
    assert all(len(entry["request"]["anchor"].split()) > 6 for entry in entries)
    assert any("development" in entry["quote"] for entry in entries)
    assert any("confirmation" in entry["quote"] for entry in entries)
    for entry in entries:
        assert_exact(entry, captured)


def test_identical_repeated_sentence_is_omitted_as_ambiguous():
    sentence = "Clipping limits transmitted updates before aggregation."
    captured = source(sentence + " " + sentence)
    assert build_excerpt_catalogue(captured) == []


def test_corrupt_pdf_controls_and_equation_header_fragments_do_not_become_options():
    captured = source("EQUATION SECTION HEADER ONLY AUTHOR DEPARTMENT.\n\n"
                      "x = y + z + q + r + t.\n\n"
                      "Clipping contains bad\x1a extraction bytes inside prose.\n\n"
                      "The control policy preserves finite recorded heating commands.")
    entries = build_excerpt_catalogue(captured)
    assert len(entries) == 1
    assert entries[0]["quote"] == "The control policy preserves finite recorded heating commands."
    assert_exact(entries[0], captured)


@pytest.mark.parametrize("updates", [
    {"sha256": "0" * 64}, {"kind": "search"}, {"metadata": {"status": "discovered"}},
    {"uri": "https://example.org/blog/paper"}, {"uri": "https://proceedings.mlr.press.evil.example/v162/paper.pdf"},
    {"id": "../unsafe"}, {"id": None},
])
def test_ineligible_or_unbound_capture_cannot_fill_the_catalogue(updates):
    captured = source("Clipping limits transmitted updates before aggregation.", **updates)
    assert build_excerpt_catalogue(captured) == []


def test_duplicate_identity_conflict_rejected_and_same_url_deduplicated_stably():
    first = source("Clipping limits transmitted updates before aggregation.")
    conflict = source("The control policy preserves finite recorded heating commands.")
    assert build_excerpt_catalogue([first, conflict]) == []
    duplicate = {**deepcopy(first), "id": "W002"}
    assert build_excerpt_catalogue([duplicate, first]) == build_excerpt_catalogue([first, duplicate])
    assert {entry["source_id"] for entry in build_excerpt_catalogue([first, duplicate])} == {"W001"}
    assert build_excerpt_catalogue([first, deepcopy(first)]) == build_excerpt_catalogue(first)


def test_discovery_same_url_cannot_shadow_eligible_extracted_capture():
    discovery = source("This search snippet is discovery material only.", identifier="W000", kind="search")
    extracted = source("Clipping limits transmitted updates before aggregation.", identifier="W001")
    assert {entry["source_id"] for entry in build_excerpt_catalogue([discovery, extracted])} == {"W001"}


def test_question_ranking_reaches_late_passages_and_keeps_beginning_context():
    text = " ".join(f"Ordinary controller report number {index} describes stable measured operation." for index in range(70))
    text += " Critic federation changes the resulting plasma transport behavior."
    captured = source(text)
    entries = build_excerpt_catalogue(captured, "Does critic federation change plasma transport?")
    assert any("report number 0" in entry["quote"] for entry in entries)
    assert any("Critic federation" in entry["quote"] for entry in entries)
    assert len(entries) <= MAX_PER_SOURCE


def test_ids_are_independent_of_question_order_but_bound_to_hash_and_source_identity():
    captured = source("Clipping limits transmitted updates before aggregation. The control policy preserves finite recorded heating commands.")
    first = build_excerpt_catalogue(captured, "clipping")
    second = build_excerpt_catalogue(captured, "heating")
    assert {entry["id"] for entry in first} == {entry["id"] for entry in second}
    renamed = build_excerpt_catalogue({**captured, "id": "W002"}, "clipping")
    assert {entry["id"] for entry in first}.isdisjoint(entry["id"] for entry in renamed)
    changed = source(captured["content"] + " Additional context describes experimental limitations and remaining uncertainty.")
    assert {entry["id"] for entry in first}.isdisjoint(entry["id"] for entry in build_excerpt_catalogue(changed))


def test_catalogue_global_cap_is_fair_across_sources_and_input_order():
    captured = [source(" ".join(f"Controller report number {index} describes stable measured operation in reactor {paper}." for index in range(20)),
                       identifier=f"W{paper:03d}", uri=f"https://arxiv.org/abs/2601.{paper:05d}") for paper in range(1, 9)]
    entries = build_excerpt_catalogue(captured)
    assert len(entries) == MAX_TOTAL
    assert len({entry["source_id"] for entry in entries}) == 8
    assert entries == build_excerpt_catalogue(list(reversed(captured)))
    assert len({entry["id"] for entry in entries}) == len(entries)


@pytest.mark.parametrize("question", [None, [], "q" * 4001])
def test_question_input_is_bounded_and_native(question):
    with pytest.raises(ValueError, match="question"):
        build_excerpt_catalogue([], question)


def test_five_word_complete_sentence_is_not_dropped():
    captured = source("Clipping can limit large updates.")
    entries = build_excerpt_catalogue(captured)
    assert len(entries) == 1
    assert entries[0]["quote"] == captured["content"]
    assert_exact(entries[0], captured)


def test_repeated_content_has_bounded_anchor_attempts(monkeypatch):
    from fusion_agent.research import grounded_excerpts as module

    calls = []
    original = module._unique_request

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(module, "_unique_request", counted)
    captured = source("Clipping limits transmitted updates before aggregation. " * 500)
    assert module.build_excerpt_catalogue(captured) == []
    assert len(calls) == module.MAX_CANDIDATE_ATTEMPTS
