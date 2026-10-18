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
_BODY_START = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:Abstract|(?:\d+(?:\.\d+)*[.)]?[ \t]+)?Introduction)"
    r"[ \t]*[:.—-]?[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)
_BODY_END = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\d+(?:\.\d+)*[.)]?[ \t]+)?"
    r"(?:References|Bibliography|Cite this (?:paper|article)|Citation|BibTeX|Endnote|RIS)"
    r"[ \t]*:?[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)
_NON_BODY_LINE = re.compile(
    r"^[ \t]*(?:\[[^\]\n]+\](?:\([^\n]*\))?|Proceedings\b[^\n]*|Copyright\b[^\n]*|©[^\n]*|"
    r"(?:DOI|ISBN|ISSN)[ \t]*:[^\n]*|(?:Download|View|Read)[ \t]+(?:PDF|HTML|paper|article)\b[^\n]*|"
    r"Home|About|Search|Edit|Back|PDF)[ \t]*\r?$", re.IGNORECASE | re.MULTILINE)
_BIBLIOGRAPHIC_FRAGMENT = re.compile(
    r"\b(?:Proceedings of|PMLR\s+\d+\s*:|ISBN|ISSN|BibTeX|Endnote|arXiv preprint)\b", re.IGNORECASE)


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
    if _BIBLIOGRAPHIC_FRAGMENT.search(quote):
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


def _body_regions(content):
    """Exclude identifiable page furniture, not judge scientific relevance.

    A recognized abstract/introduction heading bounds front matter. References
    and citation/export sections are deliberately omitted even if their titles
    overlap the question. No text is rewritten, so offsets remain canonical.
    Captures without headings retain prose outside identifiable metadata lines.
    """
    heading = _BODY_START.search(content)
    start = heading.end() if heading else 0
    ending = _BODY_END.search(content, start)
    stop = ending.start() if ending else len(content)
    regions, cursor = [], start
    for excluded in _NON_BODY_LINE.finditer(content, start, stop):
        if cursor < excluded.start():
            regions.append((cursor, excluded.start()))
        cursor = excluded.end()
    if cursor < stop:
        regions.append((cursor, stop))
    return regions


def _spans(content, question_terms):
    """Scan bounded scientific-text regions, preserving literal offsets."""
    result = []
    for begin, stop in _body_regions(content):
        cursor = begin
        ends = [match.end() for match in _SENTENCE_END.finditer(content, begin, stop)]
        if not ends or ends[-1] != stop:
            ends.append(stop)
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
                start, end = selected[0].start(), selected[-1].end()
                quote = content[start:end]
                if not _prose(quote):
                    continue
                overlap = _terms(quote) & question_terms
                result.append({"start": start, "end": end, "quote": quote,
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


def _single_match(normalized, needle):
    first = normalized.find(needle)
    return first >= 0 and normalized.find(needle, first + 1) < 0


def _normalized_body(content):
    # The separator prevents matches spanning two regions separated by metadata.
    return "\x00".join(" " + " ".join(content[start:end].split()) + " "
                         for start, end in _body_regions(content))


def resolve_catalogue_citation(source, anchor, max_words, locator=None):
    """Resolve an exact helper citation, optionally disambiguating body/export copies.

    A selected locator is never trusted as an arbitrary character range. It must
    be an exact candidate returned by the existing helper, lie inside retained
    body text, and have an anchor unique within that body. Thus identical body
    occurrences remain ambiguous; a repeated BibTeX abstract cannot suppress
    the actual abstract. No source or quotation is rewritten.
    """
    citation = cite_web_excerpt(source, anchor, max_words)
    if locator is None:
        if "candidates" in citation:
            raise ValueError("Ambiguous literature anchor; select a longer unique anchor from the stored source")
        return citation
    if not isinstance(locator, str) or not re.fullmatch(r"chars:\d{1,9}:\d{1,9}", locator):
        raise ValueError("A selected literature locator must identify an exact helper candidate")
    options = [item["citation"] for item in citation["candidates"]] if "candidates" in citation else [citation]
    matching = [item for item in options if item["locator"] == locator]
    if len(matching) != 1:
        raise ValueError("The selected literature locator is not an exact returned helper candidate")
    _, start, end = locator.split(":")
    content = source["content"]
    if not any(begin <= int(start) < int(end) <= stop for begin, stop in _body_regions(content)):
        raise ValueError("The selected literature locator is outside retained scientific text")
    needle = " " + " ".join(anchor.split()) + " "
    if not _single_match(_normalized_body(content), needle):
        raise ValueError("Ambiguous literature anchor within retained scientific text")
    return matching[0]


def _unique_request(source, candidate, normalized, normalized_body=None):
    words = candidate["quote"].split()
    # A short shared prefix is extended, never guessed or disambiguated by an
    # invented offset. If no unique anchor fits the API, omit that alternative.
    for count in range(min(6, len(words)), len(words) + 1):
        anchor = " ".join(words[:count])
        if len(anchor) > 300:
            break
        needle = " " + anchor + " "
        if _single_match(normalized, needle):
            return {"source_id": source["id"], "anchor": anchor, "max_words": len(words)}
        if normalized_body is not None and _single_match(normalized_body, needle):
            return {"source_id": source["id"], "anchor": anchor, "max_words": len(words),
                    "locator": f"chars:{candidate['start']}:{candidate['end']}"}
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
        normalized_body = _normalized_body(content)
        options, attempts = [], 0
        for candidate in _rank(_spans(content, question_terms))[:MAX_CANDIDATE_ATTEMPTS]:
            request = _unique_request(source, candidate, normalized, normalized_body)
            if request is None:
                continue
            attempts += 1
            try:
                citation = resolve_catalogue_citation(source, request["anchor"], request["max_words"], request.get("locator"))
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
