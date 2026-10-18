"""Deterministic alternatives for exact primary-source quotations.

The catalogue contains alternatives, not a report or a relevance verdict. IDs
bind source identity, content hash and exact offsets; the question affects only
ranking. The caller must retain the same server-owned question when rebuilding
a bounded catalogue. Final publication still enforces its quotation budget.
"""

from __future__ import annotations

import hashlib
import json
import re

from .primary_sources import eligible_primary_source
from .web_citations import cite_web_excerpt

MAX_PER_SOURCE = 12
MAX_TOTAL = 48
MAX_SOURCES = 12
MAX_SOURCE_CHARS = 300_000
MAX_CITATION_ATTEMPTS = 24
MAX_CANDIDATE_ATTEMPTS = 128
_CONTEXT_CHARS = 220
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,95}\Z")
_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?][\"'”’)]*(?=\s|$)|\n[ \t]*\n")
_TERM = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset("a an and are as at be been but by can could did do does for from has have how i in into is it its may might of on or our should so than that the their them these they this to use used using was we were what when where which why will with would you your".split())  # noqa: SIM905 - compact stopword vocabulary


def _terms(text):
    return {word for word in _TERM.findall(text.casefold()) if len(word) >= 3 and word not in _STOP}


def _prose(quote):
    """Conservative readability filter, never a scientific-quality judgment."""
    words = quote.split()
    if not 5 <= len(words) <= 25 or len(quote) > 1800:
        return False
    if any(ord(char) < 32 and char not in "\n\r\t" for char in quote):
        return False
    if any(marker in quote for marker in ("http://", "https://", "www.", "\\", "$$", "∑", "∇", "⇒", "∈")):
        return False
    if sum(quote.count(char) for char in "=+{}<>|$_") > 1:
        return False
    letters = sum(char.isalpha() for char in quote)
    visible = sum(not char.isspace() for char in quote)
    if not visible or letters / visible < .72:
        return False
    lexical = [word for word in words if sum(char.isalpha() for char in word) >= 2]
    substantial = {word.casefold().strip(".,;:!?()[]\"'“”‘’") for word in words if sum(char.isalpha() for char in word) >= 3}
    if len(lexical) / len(words) < .75 or len(substantial) < 4:
        return False
    if sum(word.isupper() and len(word) > 1 for word in words) / len(words) > .5:
        return False
    return not quote.lstrip().startswith(("#", "|", "[", "```"))


def _spans(content, question_terms):
    """Scan the complete bounded snapshot, preserving its literal offsets."""
    result, cursor = [], 0
    ends = [match.end() for match in _SENTENCE_END.finditer(content)]
    if not ends or ends[-1] != len(content):
        ends.append(len(content))
    for end in ends:
        words = list(_WORD.finditer(content, cursor, end))
        cursor = end
        if len(words) < 5:
            continue
        complete = len(words) <= 25 and bool(re.search(r"[.!?][\"'”’)]*$", words[-1].group()))
        starts = [0] if len(words) <= 25 else list(range(0, len(words), 20))
        if len(words) > 25 and len(words) - starts[-1] < 5:
            starts[-1] = max(0, len(words) - 25)
        for index in dict.fromkeys(starts):
            selected = words[index:index + 25]
            start, stop = selected[0].start(), selected[-1].end()
            quote = content[start:stop]
            if not _prose(quote):
                continue
            overlap = _terms(quote) & question_terms
            result.append({"start": start, "end": stop, "quote": quote,
                           "kind": "complete_sentence" if complete else "bounded_span",
                           "score": len(overlap)})
    return result


def _rank(candidates):
    if not candidates:
        return []
    # Preserve early context, then relevant material anywhere in the paper,
    # then spread remaining choices across the source instead of only its head.
    relevant = sorted((item for item in candidates if item["score"]),
                      key=lambda item: (-item["score"], item["kind"] != "complete_sentence", item["start"]))
    spread = [candidates[(len(candidates) - 1) * i // (MAX_PER_SOURCE - 1)]
              for i in range(MAX_PER_SOURCE)]
    ordered = candidates[:2] + relevant + spread + candidates
    seen, result = set(), []
    for item in ordered:
        identity = (item["start"], item["end"])
        if identity not in seen:
            seen.add(identity)
            result.append(item)
    return result


def _unique_request(source, candidate, normalized):
    words = candidate["quote"].split()
    # A short shared prefix is extended, never guessed or disambiguated by an
    # invented offset. If no unique anchor fits the API, omit that alternative.
    for count in range(min(6, len(words)), len(words) + 1):
        anchor = " ".join(words[:count])
        if len(anchor) > 300:
            break
        needle = " " + anchor + " "
        first = normalized.find(needle)
        if first >= 0 and normalized.find(needle, first + 1) < 0:
            return {"source_id": source["id"], "anchor": anchor, "max_words": len(words)}
    return None


def _sources(sources):
    if isinstance(sources, dict):
        sources = [sources] if "content" in sources else list(sources.values())
    if not isinstance(sources, list) or len(sources) > 100:
        return []
    hashes = {}
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("id"), str):
            hashes.setdefault(source["id"], set()).add(str(source.get("sha256")))
    eligible = []
    for source in sources:
        try:
            if (not isinstance(source, dict) or not _ID.fullmatch(source.get("id", ""))
                    or len(hashes[source["id"]]) != 1
                    or not eligible_primary_source(source)
                    or len(source["content"]) > MAX_SOURCE_CHARS):
                continue
            eligible.append(source)
        except (KeyError, TypeError, ValueError, UnicodeError):
            continue
    # A discovery-only capture cannot shadow an extracted paper: eligibility
    # is applied first. Same-URL duplicates choose a stable source ID.
    seen, result = set(), []
    for source in sorted(eligible, key=lambda item: (item["uri"], item["id"], item["sha256"])):
        if source["uri"] not in seen:
            seen.add(source["uri"])
            result.append(source)
    return result[:MAX_SOURCES]


def build_excerpt_catalogue(sources, question=""):
    """Return bounded, hash-bound excerpt alternatives without modifying sources.

    Entries have id, source_id, source_sha256, quote, locator, context (before /
    after), selection_kind and request. Pass only the retained request to the
    existing publication builder. The ID is independent of question and rank.
    """
    if not isinstance(question, str) or len(question) > 4000:
        raise ValueError("Excerpt ranking requires a question of at most 4000 characters")
    question_terms = _terms(question)
    per_source = []
    for source in _sources(sources):
        content = source["content"]
        normalized = " " + " ".join(content.split()) + " "
        options, attempts = [], 0
        for candidate in _rank(_spans(content, question_terms))[:MAX_CANDIDATE_ATTEMPTS]:
            request = _unique_request(source, candidate, normalized)
            if request is None:
                continue
            attempts += 1
            try:
                citation = cite_web_excerpt(source, request["anchor"], request["max_words"])
            except (KeyError, TypeError, ValueError):
                citation = None
            locator = f"chars:{candidate['start']}:{candidate['end']}"
            if (citation and "candidates" not in citation and citation.get("quote") == candidate["quote"]
                    and citation.get("locator") == locator):
                identity = {"source_id": source["id"], "source_sha256": source["sha256"],
                            "locator": locator, "quote": citation["quote"]}
                fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
                options.append({"id": "X" + fingerprint[:32], **identity,
                    "context": {"before": content[max(0, candidate["start"] - _CONTEXT_CHARS):candidate["start"]],
                                "after": content[candidate["end"]:candidate["end"] + _CONTEXT_CHARS]},
                    "selection_kind": candidate["kind"], "request": request})
            if len(options) >= MAX_PER_SOURCE or attempts >= MAX_CITATION_ATTEMPTS:
                break
        per_source.append(options)
    # Round-robin keeps the global cap from suppressing every later source.
    result = []
    for index in range(MAX_PER_SOURCE):
        for options in per_source:
            if index < len(options):
                result.append(options[index])
                if len(result) == MAX_TOTAL:
                    return result
    return result
