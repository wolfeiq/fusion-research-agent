"""Publish recorded facts and exact excerpts without trusting model prose.

This module has no model, retrieval, worker or policy-writing capabilities. Its
inputs select evidence; they cannot provide the text of a factual conclusion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

from .citation_values import cite_experiment_value
from .grounded_facts import build_grounded_facts
from .shadow import validate_config
from .web_citations import cite_web_excerpt

SCHEMA = "fusion-grounded-publication-v1"
MAX_FACTS = 40
MAX_EXCERPTS = 8
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REASONS = {
    "diagnose_before_comparison": "Diagnose the recorded incomplete, invalid or unstable trial before a larger comparison.",
    "repair_comparison_contract": "Repair the recorded comparison contract before interpreting endpoint differences.",
    "keep_baseline": "Retain the baseline while investigating recorded trade-offs; no candidate policy is approved.",
    "repeat_varied_paired_seeds": "Repeat the frozen hypothesis across additional paired development seeds; an observed endpoint difference does not establish efficacy.",
    "inconclusive": "The recorded development comparison is inconclusive; retain the baseline.",
}
LIMITATIONS = (
    "Published facts are deterministic projections of captured records, not independent validation of simulator physics or experimental design.",
    "Literature excerpts reproduce what a captured source says; they do not establish applicability to this controller or a causal mechanism.",
    "Repeated runs of the same training seed are not independent efficacy evidence. Endpoint cost is not shots-to-competence.",
    "A recommended next run is a proposal only. No policy change, experiment launch or confirmation-data use is authorized by this report.",
)


def _content(source):
    content = source.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False)
    raise ValueError("Every publication source requires captured content")


def _registry(sources):
    if isinstance(sources, dict):
        sources = [sources] if "id" in sources else list(sources.values())
    if not isinstance(sources, list) or len(sources) > 100:
        raise ValueError("Publication sources must be a bounded source registry")
    result = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Every publication source must be an object")  # noqa: TRY004 - public schema boundary
        identifier = source.get("id")
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 512
                or any(ord(char) < 32 for char in identifier)):
            raise ValueError("Invalid publication source ID")
        if identifier in result:
            raise ValueError("Duplicate publication source ID")
        if source.get("kind") not in {"experiment", "protocol", "web", "search"}:
            raise ValueError("Unknown publication source kind")
        fingerprint = source.get("sha256")
        if (not isinstance(fingerprint, str) or not _HASH.fullmatch(fingerprint)
                or hashlib.sha256(_content(source).encode("utf-8")).hexdigest() != fingerprint):
            raise ValueError("Publication source content hash is missing or mismatched")
        if not isinstance(source.get("metadata", {}), dict):
            raise ValueError("Publication source metadata must be an object")  # noqa: TRY004 - public schema boundary
        result[identifier] = {**source, "content": _content(source)}
    return result


def _selection(values, name, maximum):
    if not isinstance(values, list) or len(values) > maximum:
        raise ValueError(f"{name} must be a list of at most {maximum} items")
    return values


def _safe_title(source):
    title = source.get("title") or source["uri"]
    if (not isinstance(title, str) or len(title) > 1000
            or any(ord(char) < 32 and char not in "\r\n\t" for char in title)):
        raise ValueError("The captured source title is not displayable")
    return " ".join(title.split())


def _source_label(card):
    provenance = card.get("provenance", {})
    origin = provenance.get("origin")
    names = {"saved_shadow_snapshot": "Saved shadow trial",
             "research_shadow_execution": "Current research-run shadow trial"}
    if origin not in names:
        raise ValueError("A published fact needs recognized source provenance")
    identifier = provenance.get("trial_id")
    suffix = f" {identifier}" if isinstance(identifier, str) and identifier else ""
    return names[origin] + suffix


def _literature(registry, requests):
    cards, seen, words_by_url = [], set(), {}
    for request in _selection(requests, "literature_requests", MAX_EXCERPTS):
        if not isinstance(request, dict) or set(request) != {"source_id", "anchor", "max_words"}:
            raise ValueError("Literature requests require exactly source_id, anchor and max_words")
        source_id = request["source_id"]
        if not isinstance(source_id, str) or source_id not in registry:
            raise ValueError("Unknown literature source ID")
        source = registry[source_id]
        citation = cite_web_excerpt(source, request["anchor"], request["max_words"])
        if "candidates" in citation:
            raise ValueError("Ambiguous literature anchor; select a longer unique anchor from the stored source")
        identity = (source_id, citation["locator"])
        if identity in seen:
            raise ValueError("Duplicate literature excerpt")
        seen.add(identity)
        # Count complete quoted tokens, not unique vocabulary: 25 repetitions
        # of one word still consume 25 words. Repeated identical excerpts from
        # duplicate captures of the same URL consume their quota only once.
        quotes = words_by_url.setdefault(source["uri"], set())
        quotes.add(" ".join(citation["quote"].split()))
        if sum(len(quote.split()) for quote in quotes) > 25:
            raise ValueError("Literature excerpts exceed 25 quoted words for one source URL")
        title = _safe_title(source)
        cards.append({
            "id": "Q" + hashlib.sha256(json.dumps([source_id, source["sha256"], citation], sort_keys=True).encode()).hexdigest()[:24],
            "kind": "attributed_source_excerpt", "title": title,
            "text": f"Source excerpt from {title}: “{citation['quote']}”",
            "quote": citation["quote"], "source_id": source_id, "uri": source["uri"],
            "source_sha256": source["sha256"], "evidence": [citation],
            "provenance": {"origin": "captured_web_page", "retrieved_at": source.get("metadata", {}).get("retrieved_at")},
            "publication_basis": "exact_attributed_excerpt",
            "scope": "Exact source wording only; relevance, interpretation and application remain hypotheses.",
        })
    return cards


def _recommendation(source, cards):
    matching = [card for card in cards if card["source_id"] == source["id"]]
    if not matching or source.get("kind") != "experiment":
        raise ValueError("Next-run source must have a validated shadow fact adapter")
    document = json.loads(_content(source))
    adapter = source.get("metadata", {}).get("adapter")
    if adapter == "shadow_execution" and document.get("schema") == "fusion-advisor-shadow-evidence-v1":
        comparison, prefix = document.get("comparison"), ["comparison"]
        if comparison is None and document.get("status") in {"failed", "timed_out", "interrupted"}:
            result = {
                "code": "diagnose_before_comparison", "text": _REASONS["diagnose_before_comparison"],
                "status": "proposal_only", "request": None, "budget": None,
                "controls": [], "source_id": source["id"], "source_sha256": source["sha256"],
                "provenance": copy.deepcopy(matching[0]["provenance"]),
                "automatic_launch": False, "automatic_apply": False,
                "publication_basis": "recorded_failed_execution_without_comparison",
                "evidence": [cite_experiment_value(source, ["status"])],
            }
            result["source_label"] = _source_label(result)
            return result
    elif adapter == "shadow_comparison":
        comparison, prefix = document, []
    else:
        raise ValueError("Next-run source requires a recognized shadow comparison adapter")
    if not isinstance(comparison, dict) or comparison.get("schema") != "fusion-shadow-comparison-v1":
        raise ValueError("Next-run source lacks a recognized comparison schema")
    recorded = comparison.get("recommendation")
    proposal = comparison.get("next_run_proposal")
    if (not isinstance(recorded, dict) or recorded.get("code") not in _REASONS
            or not isinstance(proposal, dict) or proposal.get("status") != "proposal_only"):
        raise ValueError("Unknown recorded recommendation contract")
    if any(item.get(key) is not False for item in (recorded, proposal) for key in ("automatic_apply", "automatic_launch")):
        raise ValueError("Recorded proposals must prohibit automatic actions")
    if recorded.get("efficacy_claim") is not False:
        raise ValueError("Recorded recommendation must not assert efficacy")
    request, budget = proposal.get("request"), proposal.get("budget")
    citations = [cite_experiment_value(source, prefix + ["recommendation", "code"])]
    result = {
        "code": recorded["code"], "text": _REASONS[recorded["code"]],
        "status": "proposal_only", "request": None, "budget": None,
        "controls": ["local", "reset_only", "uniform", "clipped"],
        "source_id": source["id"], "source_sha256": source["sha256"],
        "provenance": copy.deepcopy(matching[0]["provenance"]),
        "automatic_launch": False, "automatic_apply": False,
        "seed_selection_scope": "Proposed additional development seeds; check the complete development/confirmation ledger before launch.",
        "publication_basis": "validated_recorded_next_run_proposal",
        "evidence": citations,
    }
    result["source_label"] = _source_label(result)
    if request is None:
        if budget is not None:
            raise ValueError("A missing next-run request cannot have a budget")
        citations.append(cite_experiment_value(source, prefix + ["next_run_proposal", "request"]))
        return result
    if recorded["code"] in {"diagnose_before_comparison", "repair_comparison_contract"}:
        raise ValueError("The recorded diagnosis recommendation cannot authorize another trial request")
    if (comparison.get("status") != "complete" or comparison.get("execution_status") != "completed"
            or comparison.get("worker_reported_status") != "complete" or comparison.get("failures") != []
            or not any(card["kind"] == "actual_budget" for card in matching)):
        raise ValueError("A next-run request requires a complete, accounted shadow comparison")
    if not isinstance(request, dict) or validate_config(request) != request or not isinstance(budget, dict):
        raise ValueError("Invalid typed next-run request or budget")
    count, rounds = len(request["seeds"]), request["rounds"]
    expected = {
        "configurations": 2, "training_episodes": count * (16 + 16 * rounds),
        "donor_training_episodes": 16 * count, "target_training_episodes": 16 * count * rounds,
        "evaluation_episodes": count * (4 + 20 * rounds),
        "initial_and_fixed_evaluation_episodes": 4 * count,
        "gate_episodes": 8 * count * rounds, "report_episodes": 12 * count * rounds,
        "total_episodes": count * (20 + 36 * rounds),
        "maximum_windows": count * (20 + 36 * rounds) * 30,
        "maximum_wall_seconds": 600, "cpu_threads": 1,
    }
    if any(type(budget.get(key)) is not int or budget[key] != value for key, value in expected.items()):
        raise ValueError("Recorded next-run budget does not match the fixed shadow protocol")
    result.update(request=copy.deepcopy(request), budget=expected,
                  budget_scope="Gate, initial/fixed and report episodes are disjoint subsets of evaluation, not additional episodes. Prior calibration is excluded; optimizer updates are separate work.")
    citations.extend(cite_experiment_value(source, prefix + ["next_run_proposal", field]) for field in ("request", "budget"))
    return result


def build_publication(sources, fact_ids, literature_requests, next_run_source_id=None):
    """Compile source-bound output; model-authored conclusions are not an input."""
    registry = _registry(sources)
    all_facts = build_grounded_facts(list(registry.values()))
    lookup = {card["id"]: card for card in all_facts}
    if len(lookup) != len(all_facts):
        raise ValueError("Grounded fact IDs must be unique")
    selected, seen = [], set()
    for identifier in _selection(fact_ids, "fact_ids", MAX_FACTS):
        if not isinstance(identifier, str) or identifier not in lookup:
            raise ValueError("Unknown grounded fact ID; select an existing server fact")
        if identifier in seen:
            raise ValueError("Duplicate grounded fact ID")
        seen.add(identifier)
        card = copy.deepcopy(lookup[identifier])
        card["source_label"] = _source_label(card)
        selected.append(card)
    excerpts = _literature(registry, literature_requests)
    recommendation = None
    if next_run_source_id is not None:
        if not isinstance(next_run_source_id, str) or next_run_source_id not in registry:
            raise ValueError("Unknown next-run source ID")
        recommendation = _recommendation(registry[next_run_source_id], all_facts)
    paragraphs = [f"{card['source_label']}: {card['text']} [{card['source_id']}]" for card in selected]
    paragraphs.extend(f"{card['text']} [{card['source_id']}]" for card in excerpts)
    if recommendation:
        paragraphs.append(f"{recommendation['source_label']} — next-run proposal: {recommendation['text']} [{recommendation['source_id']}]")
    grounded = bool(selected and excerpts)
    if not grounded:
        paragraphs.insert(0, "Insufficient evidence for both a grounded experiment finding and an attributed literature excerpt.")
    return {
        "schema": SCHEMA, "status": "grounded" if grounded else "insufficient_evidence",
        "facts": selected, "literature_excerpts": excerpts,
        "recommendation": recommendation, "limitations": list(LIMITATIONS),
        "answer": "\n\n".join(paragraphs),
        "scientific_efficacy_established": False,
        "automatic_policy_changes_allowed": False,
    }
