"""The judge sees each claim's cited scope, not convenient facts elsewhere."""

import hashlib
import json
from copy import deepcopy

import pytest

from fusion_agent.research.review_evidence import review_payload


def source(identifier, kind, data):
    content = json.dumps(data) if kind == "experiment" else data
    return {"id": identifier, "kind": kind, "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "uri": "https://example.org/paper" if kind == "web" else "fixture.json"}


def claim(identifier, source_id, locator, quote, kind="observation"):
    return {"id": identifier, "kind": kind, "text": "Unverified claim under review.",
            "evidence": [{"source_id": source_id, "locator": locator, "quote": quote}]}


def draft(claims, next_runs=None):
    return {"claims": claims, "next_runs": next_runs or []}


def test_same_pointer_has_same_packet_and_cannot_expose_uncited_roles_or_baselines():
    candidate = {"configuration": "candidate", "role": "deposition_sac", "clipped_count": 2}
    saved = source("E001", "experiment", {
        "records": [{"baseline_td3_fact": "UNCITED_BASELINE"}, candidate],
        "uncited_endpoint": 998877.123,
    })
    original = deepcopy(saved)
    citation = claim("C1", "E001", "/records/1", json.dumps(candidate, sort_keys=True))
    control = {**deepcopy(citation), "id": "CONTROL", "text": "A differently scoped claim."}
    payload = review_payload(draft([citation, control]), [saved], "Review the citations.")
    first, second = [item["evidence"][0] for item in payload["claims"]]
    assert first == second
    assert first["cited_json_value"] == candidate
    assert first["canonical_sha256"] == saved["sha256"]
    assert "UNCITED_BASELINE" not in json.dumps(payload)
    assert "998877.123" not in json.dumps(payload)
    assert saved == original


def test_difference_scope_and_proposal_ids_cannot_supply_uncited_absolute_costs():
    saved = source("E001", "experiment", {
        "differences": {"clipped": {"difference": -0.001}, "local": {"difference": 0.0}},
        "absolute_endpoints": {"candidate": 0.096265, "baseline": 0.096289},
    })
    proposal_only = source("E002", "experiment", {"source_training_episodes": 48})
    citation = claim("C2", "E001", "/differences/clipped", '{"difference": -0.001}')
    payload = review_payload(draft([citation], [{"evidence_ids": ["E001", "E002"]}]),
                             [saved, proposal_only], "Review.")
    packet = payload["claims"][0]["evidence"][0]
    assert packet["cited_json_value"] == {"difference": -0.001}
    assert all("content" not in item for item in payload["sources"])
    assert "0.096265" not in json.dumps(payload)
    assert "source_training_episodes" not in json.dumps(payload)


def test_web_packet_honors_exact_occurrence_and_exposes_local_scope_not_abstract():
    quote = "Clipping sets the privacy noise scale."
    content = ("Client-level FL abstract SECRET_ABSTRACT. " + quote + "\n" + "Unrelated text. " * 100
               + "The common mechanism in centralized training clips stochastic gradients. "
               + quote + " However, FL privacy differs from centralized training. " + "Following text. " * 100)
    saved = source("W001", "web", content)
    original = deepcopy(saved)
    start = content.rindex(quote)
    citation = claim("C4", "W001", f"chars:{start}:{start + len(quote)}", quote, "literature")
    payload = review_payload(draft([citation]), [saved], "Check scope.")
    packet = payload["claims"][0]["evidence"][0]
    excerpt = packet["excerpts"][0]
    assert excerpt["quote_start"] == start and excerpt["quote_end"] == start + len(quote)
    assert excerpt["text"] == content[excerpt["context_start"]:excerpt["context_end"]]
    assert "centralized training clips stochastic gradients" in excerpt["text"]
    assert "FL privacy differs" in excerpt["text"]
    assert "SECRET_ABSTRACT" not in json.dumps(payload)
    assert packet["canonical_sha256"] == saved["sha256"]
    assert saved == original


def test_ambiguous_legacy_web_quote_exposes_multiple_contexts_and_omission_count():
    quote = "Repeated exact words."
    content = "\n".join(f"Context {number}. {quote}" for number in range(4))
    saved = source("W001", "web", content)
    packet = review_payload(draft([claim("C1", "W001", "", quote, "literature")]), [saved], "Review.")["claims"][0]["evidence"][0]
    assert packet["legacy_locator_ambiguous"] is True
    assert packet["matching_occurrences"] == 4 and packet["omitted_occurrences"] == 1
    assert len(packet["excerpts"]) == 3
    for excerpt in packet["excerpts"]:
        assert content[excerpt["quote_start"]:excerpt["quote_end"]] == quote


def test_bad_web_offset_never_falls_back_to_a_matching_quote_elsewhere():
    saved = source("W001", "web", "A real quotation.")
    packet = review_payload(draft([claim("C1", "W001", "chars:99:115", "A real quotation.", "literature")]), [saved], "Review.")["claims"][0]["evidence"][0]
    assert "evidence_unavailable" in packet and "excerpts" not in packet


def test_large_cited_json_value_is_explicitly_omitted_without_partial_json_or_parent():
    saved = source("E001", "experiment", {"large": {"quoted": "recorded", "padding": "x" * 9000}})
    packet = review_payload(draft([claim("C1", "E001", "/large", '"recorded"')]), [saved], "Review.")["claims"][0]["evidence"][0]
    assert "evidence_unavailable" in packet and "cited_json_value" not in packet
    assert packet["quote"] == '"recorded"'


@pytest.mark.parametrize("locator", ["chars:99:115", "chars:0:1", "section:1"])
def test_protocol_span_must_match_actual_quote_even_when_legacy_verifier_is_text_only(locator):
    saved = source("P001", "protocol", "A real protocol quotation.")
    packet = review_payload(draft([claim("C1", "P001", locator, "A real protocol quotation.")]), [saved], "Review.")["claims"][0]["evidence"][0]
    assert "evidence_unavailable" in packet and "excerpts" not in packet


def test_web_packet_retains_known_capture_truncation_warning():
    saved = source("W001", "web", "A captured paragraph.")
    saved["metadata"] = {"truncated": True, "uncited_note": "DO_NOT_LEAK"}
    packet = review_payload(draft([claim("C1", "W001", "", saved["content"], "literature")]), [saved], "Review.")["claims"][0]["evidence"][0]
    assert packet["captured_extraction_truncated"] is True
    assert "may already be a truncated extraction" in packet["context_notice"]
    assert "DO_NOT_LEAK" not in json.dumps(packet)
