"""Select a short contiguous quotation from a stored, extracted public page."""

from __future__ import annotations

import re

from .providers import public_url


def cite_web_excerpt(source, anchor, max_words):
    if (not isinstance(source, dict) or source.get("kind") != "web"
            or source.get("metadata", {}).get("status") != "extracted"):
        raise ValueError("Choose an already-extracted registered web source")
    public_url(source.get("uri"))
    content = source.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The extracted source has no stored text")
    if not isinstance(anchor, str) or not anchor.strip() or len(anchor) > 300:
        raise ValueError("anchor must contain 1 to 300 characters from the stored page")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in anchor):
        raise ValueError("anchor contains unsupported control characters")
    if type(max_words) is not int or not 1 <= max_words <= 25:
        raise ValueError("max_words must be an integer from 1 to 25")
    normalized_anchor = " ".join(anchor.split())
    if len(normalized_anchor.split()) > max_words:
        raise ValueError("The entire anchor must fit within max_words")

    # Map whitespace-normalized text back to original character offsets. No
    # case folding, punctuation rewriting, fuzzy matching or generated ellipses.
    parts, offsets = [], []
    previous_end = 0
    for match in re.finditer(r"\S+", content):
        if parts:
            parts.append(" ")
            offsets.append(previous_end)
        parts.append(match.group())
        offsets.extend(range(match.start(), match.end()))
        previous_end = match.end()
    normalized = "".join(parts)
    positions = []
    cursor = 0
    while (position := normalized.find(normalized_anchor, cursor)) >= 0:
        following = position + len(normalized_anchor)
        if ((position == 0 or normalized[position - 1] == " ")
                and (following == len(normalized) or normalized[following] == " ")):
            positions.append(position)
        cursor = position + 1
    if not positions:
        raise ValueError("The exact anchor was not found in the stored page")
    candidates = []
    rejected_candidates = []
    for position in positions[:5]:
        start = offsets[position]
        anchor_end = offsets[position + len(normalized_anchor) - 1] + 1
        words = list(re.finditer(r"\S+", content[start:]))[:max_words]
        end = start + words[-1].end()
        if anchor_end > end:
            raise ValueError("The entire anchor must fit within max_words")
        quote = content[start:end]
        if len(quote) > 2000:
            raise ValueError("Excerpt exceeds 2000 characters; use a smaller word budget or another anchor")
        forbidden = sorted({ord(char) for char in quote if ord(char) < 32 and char not in "\n\r\t"})
        if forbidden:
            # PDF extraction can insert non-display controls even when the
            # anchor itself is clean. Never hand back a citation the report
            # validator will reject, or silently rewrite immutable source text.
            rejected_candidates.append({
                "locator": f"chars:{start}:{end}",
                "reason": "unsupported control characters",
                "codepoints": [f"U+{codepoint:04X}" for codepoint in forbidden],
            })
            continue
        candidates.append({
            "citation": {"source_id": source["id"], "locator": f"chars:{start}:{end}", "quote": quote},
            "context_before": content[max(0, start - 160):start],
            "context_after": content[end:end + 160],
        })
    if not candidates:
        codepoints = sorted({codepoint for item in rejected_candidates for codepoint in item["codepoints"]})
        raise ValueError(
            "Examined excerpts contain unsupported control characters "
            f"({', '.join(codepoints)}); choose a clean prose anchor or a smaller word budget. "
            "The stored source has not been altered."
        )
    if len(positions) == 1:
        return candidates[0]["citation"]
    result = {
        "candidates": candidates, "total_matches": len(positions),
        "omitted_matches": max(0, len(positions) - len(candidates) - len(rejected_candidates)),
        "instruction": "The anchor occurs more than once. Choose one candidate using its source context and copy only its citation object unchanged; do not join candidates.",
    }
    if rejected_candidates:
        result["rejected_candidates"] = rejected_candidates
        result["instruction"] += (
            " Occurrences containing unsupported control characters were rejected; "
            "choose a listed clean citation or use a clean prose anchor or smaller word budget."
        )
    return result
