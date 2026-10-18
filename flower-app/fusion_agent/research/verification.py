"""Validate a proposed report and check citation *traceability*, not its truth.

Matching a quotation to a source cannot establish that the source supports the
claim: a model can quote a number and invert its meaning. A separate semantic
review must examine every factual claim. This module never launches proposals,
executes commands, fetches a URL, or changes experiment state.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any
from urllib.parse import urlsplit

MAX_CLAIMS = 12
MAX_EVIDENCE_PER_CLAIM = 16
MAX_PROPOSALS = 4
MAX_WEB_QUOTE_WORDS = 25
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,95}\Z")
_COMMAND = re.compile(
    r"```|(?:^|\n)\s*(?:[$>]\s*)?(?:"
    r"(?:python(?:3)?|pip(?:3)?|uv|bash|sh|cmd|powershell|pwsh|curl|wget)\s+"
    r"|kaggle\s+(?:kernels|datasets)\s+|(?:sudo\s+)?(?:rm|chmod)\s+"
    r"|(?:Start-Process|Remove-Item|Invoke-Expression)\b)",
    re.IGNORECASE,
)


def _string(value: Any, name: str, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")  # noqa: TRY004 - public schema API
    value = value.strip()
    if len(value) > limit or (not value and not empty):
        raise ValueError(
            f"{name} must contain {'0' if empty else '1'}–{limit} characters"
        )
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _identifier(value: Any, name: str) -> str:
    value = _string(value, name, 96)
    if not _ID.fullmatch(value):
        raise ValueError(f"{name} is not a valid identifier")
    return value


def _list(value: Any, name: str, maximum: int, minimum: int = 0) -> list:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        received = type(value).__name__
        if isinstance(value, (list, str, dict)):
            received += f" of length {len(value)}"
        raise ValueError(
            f"{name} must be a list of {minimum}–{maximum} items; received {received}. "
            "Return a native JSON array within the stated bounds, not a serialized JSON string."
        )
    return value


def _strings(value: Any, name: str, maximum: int, minimum: int = 0) -> list[str]:
    return [
        _string(item, f"{name}[{index}]", 1500)
        for index, item in enumerate(_list(value, name, maximum, minimum))
    ]


def _verbatim_prose(value: Any, name: str, limit: int) -> str:
    """Validate free prose without deleting even its surrounding whitespace."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")  # noqa: TRY004 - public schema API
    if not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must contain 1–{limit} characters of nonblank prose")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def normalize_report_transport(report: dict) -> tuple[dict, list[dict]]:
    """Adapt only the two known limitations prose shapes, never scientific fields.

    The advertised schema remains canonical. This boundary adapter accepts a
    single report-level paragraph or a bounded native list of proposal-level
    paragraphs. Every character and item order survive; metadata records the
    original boundaries. All other schema and evidence checks run afterwards.
    """
    result = report.copy()
    changes = []
    limitations = report.get("limitations")
    if isinstance(limitations, str):
        _verbatim_prose(limitations, "limitations", 1500)
        # An encoded container is not a single paragraph. Do not reinterpret
        # nested JSON serialization as prose or decode it into more arguments.
        try:
            encoded = json.loads(limitations)
        except ValueError:
            encoded = None
        if isinstance(encoded, (dict, list)):
            raise ValueError("limitations must be prose, not a serialized JSON container")  # noqa: TRY004 - public schema API
        result["limitations"] = [limitations]
        changes.append({
            "path": "/limitations", "source_type": "string", "target_type": "array",
            "item_count": 1, "source_item_lengths": [len(limitations)],
        })

    proposals = report.get("next_runs")
    if isinstance(proposals, list) and len(proposals) <= MAX_PROPOSALS:
        result["next_runs"] = proposals.copy()
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, dict) or not isinstance(proposal.get("limitations"), list):
                continue
            name = f"next_runs[{index}].limitations"
            paragraphs = _list(proposal["limitations"], name, 12, 1)
            for item_index, paragraph in enumerate(paragraphs):
                _verbatim_prose(paragraph, f"{name}[{item_index}]", 2500)
            joined = "\n\n".join(paragraphs)
            _verbatim_prose(joined, name, 2500)
            result["next_runs"][index] = {**proposal, "limitations": joined}
            changes.append({
                "path": f"/next_runs/{index}/limitations", "source_type": "array",
                "target_type": "string", "item_count": len(paragraphs),
                "separator": "\n\n", "source_item_lengths": [len(item) for item in paragraphs],
            })
    return result, changes


def _object(value: Any, name: str, keys: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")  # noqa: TRY004 - public schema API
    if set(value) != keys:
        raise ValueError(
            f"{name} requires exactly these fields: {', '.join(sorted(keys))}"
        )
    return value


def _normalise(value: str) -> str:
    return " ".join(value.split())


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an exact RFC 6901 pointer, rejecting malformed escapes/indexes."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("Experiment locator must be an exact JSON pointer")
    current = document
    for raw_token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw_token):
            raise ValueError("Invalid JSON pointer escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            index = int(token)
            if index >= len(current):
                raise ValueError("JSON pointer array index does not exist")
            current = current[index]
        else:
            raise ValueError("JSON pointer does not resolve to a recorded value")
    return current


def _content(source: dict) -> str:
    value = source.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return ""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value}")


def _trace_evidence(evidence: dict, source: dict | None) -> str | None:
    if source is None:
        return "The source ID is not in the collected evidence."
    kind = source.get("kind")
    if not isinstance(kind, str):
        return "This source kind cannot support a factual claim."
    if kind == "search":
        return "A search snippet is not an extracted, read web page."
    if kind not in {"web", "experiment", "protocol"}:
        return "This source kind cannot support a factual claim."
    quote = _normalise(evidence["quote"])
    if not quote:
        return "A nonempty verbatim quotation is required."
    content = _content(source)
    if not content.strip():
        return "The source has no read content."
    if kind == "web":
        try:
            uri = urlsplit(source.get("uri", ""))
            valid = (
                uri.scheme in {"http", "https"}
                and bool(uri.hostname)
                and uri.username is None
                and uri.password is None
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            return "An extracted web source needs an HTTP(S) citation URL."
        # Empty locators are the legacy whole-page citation format. An explicit
        # offset locator promises an exact span, so it must not fall back to a
        # matching quotation elsewhere on the page.
        locator = evidence["locator"]
        if locator:
            span = re.fullmatch(r"chars:(0|[1-9][0-9]*):(0|[1-9][0-9]*)", locator)
            if span is None:
                return "A web locator must be empty or an exact chars:start:end span."
            start, end = map(int, span.groups())
            if not 0 <= start < end <= len(content):
                return "The web citation character span is outside the stored source."
            if content[start:end] != evidence["quote"]:
                return "The quotation does not exactly match the cited web character span."
            content = content[start:end]
    if kind == "experiment":
        # An empty RFC pointer technically selects the whole document, but it
        # defeats a judge's ability to locate the exact result being cited.
        if not evidence["locator"].startswith("/"):
            return "An experiment claim needs a non-root JSON pointer."
        try:
            document = json.loads(content, parse_constant=_reject_json_constant)
            value = resolve_json_pointer(document, evidence["locator"])
            content = json.dumps(
                value, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        except (TypeError, ValueError, OverflowError):
            return "The experiment JSON pointer does not resolve."
    if quote not in _normalise(content):
        return "The quotation does not occur at the cited source location."
    return None


def render_answer(claims: list[dict]) -> str:
    """Construct an answer without trusting free-form uncited model prose.

    Unsupported claims remain available in the claim table for inspection, but
    are excluded here. A later semantic reviewer can downgrade a claim and call
    this function again. 'Traceable' is deliberately never labelled 'verified'.
    """
    paragraphs = []
    for claim in claims:
        status = claim.get("verification", {}).get("status")
        if status == "traceable":
            ids = list(dict.fromkeys(item["source_id"] for item in claim["evidence"]))
            citations = " ".join(f"[{source_id}]" for source_id in ids)
            paragraphs.append(f"{claim['text']} {citations}")
        elif status == "hypothesis":
            paragraphs.append(f"Hypothesis — not established: {claim['text']}")
    if not any(
        claim.get("verification", {}).get("status") == "traceable" for claim in claims
    ):
        paragraphs.insert(0, "Insufficient traceable evidence for a factual answer.")
    return "\n\n".join(paragraphs)


def validate_report(report: dict, sources: list[dict]) -> dict:
    """Return a bounded, whitelisted report with deterministic citation checks.

    Invalid schema raises ValueError. Missing, unread, mismatched or excessive
    quotations downgrade a factual claim to 'unsupported' rather than silently
    accepting it. Both true and false claims can pass structural checks; callers
    must perform semantic review before presenting a final research verdict.
    """
    report = _object(
        report,
        "report",
        {"answer", "claims", "next_runs", "limitations", "stop_reason"},
    )
    _string(report["answer"], "answer", 30000, empty=True)
    source_map = {}
    for source in _list(sources, "sources", 100):
        if not isinstance(source, dict):
            raise ValueError("Every source must be an object")  # noqa: TRY004
        identifier = _identifier(source.get("id"), "source.id")
        if identifier in source_map:
            raise ValueError("Source IDs must be unique")
        source_map[identifier] = source

    claims = []
    identifiers = set()
    web_quotes = defaultdict(set)
    for raw in _list(report["claims"], "claims", MAX_CLAIMS, 1):
        raw = _object(raw, "claim", {"id", "text", "kind", "confidence", "evidence"})
        identifier = _identifier(raw["id"], "claim.id")
        if identifier in identifiers:
            raise ValueError("Claim IDs must be unique")
        identifiers.add(identifier)
        kind = _string(raw["kind"], "claim.kind", 32)
        confidence = _string(raw["confidence"], "claim.confidence", 32)
        if kind not in {"observation", "literature", "hypothesis"}:
            raise ValueError("Unknown claim kind")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("Unknown claim confidence")
        claim = {
            "id": identifier,
            "text": _string(raw["text"], "claim.text", 2000),
            "kind": kind,
            "confidence": "low" if kind == "hypothesis" else confidence,
            "evidence": [],
        }
        for raw_evidence in _list(raw["evidence"], "claim.evidence", MAX_EVIDENCE_PER_CLAIM):
            raw_evidence = _object(
                raw_evidence, "evidence", {"source_id", "quote", "locator"}
            )
            evidence = {
                "source_id": _identifier(raw_evidence["source_id"], "source_id"),
                "quote": _string(raw_evidence["quote"], "quote", 2000, empty=True),
                "locator": _string(raw_evidence["locator"], "locator", 512, empty=True),
            }
            claim["evidence"].append(evidence)
            source = source_map.get(evidence["source_id"])
            if source is not None and source.get("kind") in ("web", "search"):
                web_quotes[evidence["source_id"]].add(_normalise(evidence["quote"]))
        claims.append(claim)

    excessive_sources = {
        source_id
        for source_id, quotes in web_quotes.items()
        if sum(len(quote.split()) for quote in quotes) > MAX_WEB_QUOTE_WORDS
    }
    for claim in claims:
        failures = []
        for evidence in claim["evidence"]:
            source_id = evidence["source_id"]
            if source_id in excessive_sources:
                # Do not return copyrighted overflow quotes even on failed claims.
                evidence["quote"] = ""
                failure = (
                    "Quoted words exceed the 25-word report limit for this source."
                )
            else:
                failure = _trace_evidence(evidence, source_map.get(source_id))
            evidence["verification"] = {
                "status": "unsupported" if failure else "traceable",
                "reason": failure or "Quotation matches collected source content.",
            }
            if failure:
                failures.append(f"{source_id}: {failure}")
        if claim["kind"] == "hypothesis":
            verification = {
                "status": "hypothesis",
                "reason": "A proposed explanation to test, not an established finding.",
            }
        elif not claim["evidence"] or failures:
            verification = {
                "status": "unsupported",
                "reason": " ".join(failures)
                or "No supporting citations were provided.",
            }
            claim["confidence"] = "low"
        else:
            verification = {
                "status": "traceable",
                "reason": (
                    "Every quotation is locatable. Semantic support, contradictions, "
                    "and the claim's interpretation still require review."
                ),
            }
        claim["verification"] = verification

    proposals = []
    for raw in _list(report["next_runs"], "next_runs", MAX_PROPOSALS):
        raw = _object(
            raw,
            "next_run",
            {
                "title",
                "hypothesis",
                "changes",
                "controls",
                "metrics",
                "budget",
                "stop_rule",
                "evidence_ids",
                "limitations",
            },
        )
        proposal = {
            key: (_verbatim_prose if key == "limitations" else _string)(
                raw[key], f"next_run.{key}", 2500
            )
            for key in (
                "title",
                "hypothesis",
                "changes",
                "budget",
                "stop_rule",
                "limitations",
            )
        }
        proposal["controls"] = _strings(raw["controls"], "controls", 12, 1)
        proposal["metrics"] = _strings(raw["metrics"], "metrics", 12, 1)
        proposal["evidence_ids"] = [
            _identifier(item, "next_run.evidence_id")
            for item in _list(raw["evidence_ids"], "evidence_ids", 16, 1)
        ]
        if any(item not in source_map for item in proposal["evidence_ids"]):
            raise ValueError("Next-run evidence IDs must refer to collected sources")
        prose = "\n".join(
            item if isinstance(item, str) else "\n".join(item)
            for item in proposal.values()
        )
        if _COMMAND.search(prose):
            raise ValueError("Next runs must be reviewable proposals, not commands")
        proposal["status"] = "proposal_only"
        proposals.append(proposal)

    statuses = [claim["verification"]["status"] for claim in claims]
    return {
        "answer": render_answer(claims),
        "claims": claims,
        "next_runs": proposals,
        "limitations": [
            _verbatim_prose(item, f"limitations[{index}]", 1500)
            for index, item in enumerate(_list(report["limitations"], "limitations", 12))
        ],
        "stop_reason": _string(report["stop_reason"], "stop_reason", 2000),
        "verification": {
            "mode": "structural_traceability_only",
            "semantic_review_required": True,
            "traceable_claims": statuses.count("traceable"),
            "unsupported_claims": statuses.count("unsupported"),
            "hypotheses": statuses.count("hypothesis"),
        },
    }
