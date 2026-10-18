"""Bounded, explicitly partial model views of immutable web snapshots.

Canonical source contents and hashes remain unchanged in the evidence store.
Views contain only exact source slices with offsets, never model summaries.
"""
from __future__ import annotations

import copy
import re


def _merge(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _length(ranges):
    return sum(end - start for start, end in _merge(ranges))


def source_views(sources, *, reference_texts=(), quotes_by_source=None, max_web_chars=8000):
    """Preserve full local records; project captured web snapshots into context.

    Offsets address the canonical captured text, not the live website. Quotation
    matching permits whitespace differences, just like deterministic citation
    checking, but every emitted excerpt remains an exact original substring.
    """
    if not isinstance(max_web_chars, int) or max_web_chars < 1200:
        raise ValueError("Web source views require at least 1200 characters")
    quoted = [quote for text in reference_texts if isinstance(text, str)
              for quote in re.findall(r'["“]([^"”\n]{12,2000})["”]', text)]
    quotes_by_source = quotes_by_source or {}
    views = []
    for source in sources:
        view = copy.deepcopy(source)
        content = source.get("content")
        if source.get("kind") != "web" or not isinstance(content, str) or len(content) <= max_web_chars:
            views.append(view)
            continue
        ranges = [(0, min(1200, len(content)))]
        missed_quotes = 0
        candidates = list(dict.fromkeys([*quotes_by_source.get(source["id"], []), *quoted]))
        for quote in candidates:
            if not isinstance(quote, str) or not quote.strip():
                continue
            pattern = r"\s+".join(re.escape(part) for part in quote.split())
            match = re.search(pattern, content)
            if not match:
                continue  # Never invent or approximate a missing passage.
            surrounding = (max(0, match.start() - 500), min(len(content), match.end() + 500))
            if _length([*ranges, surrounding]) <= max_web_chars:
                ranges = _merge([*ranges, surrounding])
            elif _length([*ranges, match.span()]) <= max_web_chars:
                ranges = _merge([*ranges, match.span()])
            else:
                missed_quotes += 1
        remaining = max_web_chars - _length(ranges)
        ranges = _merge([*ranges, (0, min(len(content), ranges[0][1] + remaining))])
        included = _length(ranges)
        view["content"] = {
            "representation": "partial_verbatim_excerpts",
            "canonical_source_id": source["id"],
            "canonical_sha256": source["sha256"],
            "canonical_characters": len(content),
            "included_characters": included,
            "omitted_characters": len(content) - included,
            "matched_quotes_not_included": missed_quotes,
            "notice": "Partial view of the captured source, which may itself be a truncated extraction. Offsets use the captured text; omitted text may contain qualifications or counterevidence. Do not infer absence from omissions. Full captured snapshots remain available for citation checks and user inspection.",
            "excerpts": [{"start": start, "end": end, "text": content[start:end]}
                         for start, end in ranges],
        }
        views.append(view)
    return views
