"""Give the semantic judge each claim's cited evidence, never whole local runs.

Canonical records remain unchanged. These packets enforce citation scope but
make no scientific verdict: the model must still judge meaning and coverage.
"""

from __future__ import annotations

import copy
import json
import re

from .verification import _content, _trace_evidence, resolve_json_pointer

MAX_VALUE_CHARACTERS = 8000
MAX_PACKET_CONTENT_CHARACTERS = 48000
CONTEXT_BEFORE = 700
CONTEXT_AFTER = 500


def _text_spans(content, citation):
    if citation["locator"]:
        match = re.fullmatch(r"chars:(0|[1-9][0-9]*):(0|[1-9][0-9]*)", citation["locator"])
        if match is None:
            return [], "The text citation locator is not a supported character span."
        start, end = map(int, match.groups())
        if not 0 <= start < end <= len(content) or content[start:end] != citation["quote"]:
            return [], "The text quotation does not exactly match the specified character span."
        return [(start, end)], None
    pattern = r"\s+".join(re.escape(part) for part in citation["quote"].split())
    matches = [match.span() for match in re.finditer(pattern, content)]
    return matches, None


def review_payload(report, sources, question):
    """Build isolated citation packets plus a content-free source registry.

Proposals remain unverified author text. A proposal-only source ID contributes
identity metadata, never background results that could rescue an uncited claim.
Large values and budget omissions are explicit; no truncated JSON is evidence.
"""
    source_map = {source["id"]: source for source in sources}
    references = {citation["source_id"] for claim in report["claims"] for citation in claim["evidence"]}
    references.update(identifier for proposal in report["next_runs"] for identifier in proposal["evidence_ids"])
    remaining = MAX_PACKET_CONTENT_CHARACTERS
    claims = []
    for claim in report["claims"]:
        packets = []
        for citation in claim["evidence"]:
            source = source_map.get(citation["source_id"])
            packet = {key: copy.deepcopy(citation[key]) for key in ("source_id", "locator", "quote")}
            packet.update(canonical_sha256=source.get("sha256") if source else None,
                          source_kind=source.get("kind") if source else None)
            failure = _trace_evidence(citation, source)
            if failure:
                packet["evidence_unavailable"] = failure
            elif source["kind"] == "experiment":
                value = resolve_json_pointer(json.loads(_content(source)), citation["locator"])
                encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                packet["support_scope"] = "Only the exact JSON value at this locator; no parent, sibling, other role, baseline or control-arm records are supplied."
                if len(encoded) <= min(MAX_VALUE_CHARACTERS, remaining):
                    packet["cited_json_value"] = value
                    remaining -= len(encoded)
                else:
                    packet["evidence_unavailable"] = "The resolved value exceeds this review's content budget. Only the submitted quote is available; request a narrower citation if coverage cannot be judged."
            else:
                content = _content(source)
                spans, error = _text_spans(content, citation)
                packet["support_scope"] = "Only this quoted passage can support the claim. Adjacent context can qualify its meaning; it cannot supply uncited claims or generalize to a different setting."
                if error:
                    packet["evidence_unavailable"] = error
                else:
                    packet["legacy_locator_ambiguous"] = len(spans) > 1
                    packet["matching_occurrences"] = len(spans)
                    packet["excerpts"] = []
                    for start, end in spans[:3]:
                        low, high = max(0, start - CONTEXT_BEFORE), min(len(content), end + CONTEXT_AFTER)
                        if high - low > remaining:
                            break
                        packet["excerpts"].append({
                            "quote_start": start, "quote_end": end,
                            "context_start": low, "context_end": high,
                            "text": content[low:high],
                        })
                        remaining -= high - low
                    packet["omitted_occurrences"] = len(spans) - len(packet["excerpts"])
                    packet["canonical_characters"] = len(content)
                    captured_truncation = source.get("metadata", {}).get("truncated")
                    packet["captured_extraction_truncated"] = captured_truncation if isinstance(captured_truncation, bool) else None
                    packet["context_notice"] = "Bounded verbatim context from the captured snapshot, which may already be a truncated extraction; omitted text may contain qualifications. An ambiguous legacy locator does not identify which occurrence was intended; request exact offsets if contexts differ."
                    if not packet["excerpts"]:
                        packet["evidence_unavailable"] = "No bounded cited passage could be supplied; request a usable exact citation rather than infer support."
            packets.append(packet)
        claims.append({"id": claim["id"], "text": claim["text"], "kind": claim["kind"], "evidence": packets})
    return {
        "question": question, "claims": claims,
        "next_runs": copy.deepcopy(report["next_runs"]),
        "sources": [{key: copy.deepcopy(source.get(key)) for key in ("id", "kind", "uri", "sha256")}
                    for source in sources if source["id"] in references],
        "evidence_contract": "Judge every clause against its own claim's evidence packets only. Source identities, other claims, question wording and next-run proposals are not supporting evidence. A proposal requiring facts outside the supplied citations needs revision; no whole-source fallback is available.",
    }
