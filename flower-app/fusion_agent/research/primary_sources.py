"""Conservative publication-location eligibility, not a scientific truth test.

Only captured, extracted and hash-bound pages at recognized paper locations
can fill the research workflow's primary-reference slot. A recognized location
does not establish authorship, peer review, methodological quality or relevance.
Unknown venues remain usable discovery material, but cannot satisfy this slot.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlsplit

PRIMARY_LOCATION_SCOPE = (
    "Literature eligibility checks a recognized publication URL and an extracted, "
    "hash-bound snapshot; it does not certify authorship, peer review, scientific "
    "correctness or applicability to this controller."
)
MISSING_PRIMARY_GAP = (
    "No eligible extracted primary-publication reference was selected. Search "
    "snippets and secondary summaries cannot satisfy the primary-source requirement."
)


def approved_primary_url(value: object) -> bool:
    """Match exact publication hosts and paper paths, never title keywords."""
    if not isinstance(value, str) or any(ord(char) < 33 for char in value):
        return False
    try:
        url = urlsplit(value)
        if (url.scheme != "https" or url.username is not None or url.password is not None
                or url.port not in (None, 443) or not url.hostname):
            return False
    except ValueError:
        return False
    host, path = url.hostname.lower(), url.path
    if "%" in path or ".." in path or "\\" in path:
        return False
    patterns = {
        "proceedings.mlr.press": r"/v\d+/[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)?\.(?:html|pdf)",
        "arxiv.org": r"/(?:abs|pdf|html)/(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?(?:\.pdf)?",
        "jmlr.org": r"/(?:papers/(?:v\d+|volume\d+)/(?:[A-Za-z0-9_-]+/)?[A-Za-z0-9_-]+\.(?:html|pdf)|papers/volume\d+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\.pdf)",
        "www.jmlr.org": r"/papers/(?:v\d+|volume\d+)/(?:[A-Za-z0-9_-]+/)?[A-Za-z0-9_-]+\.(?:html|pdf)",
        "ojs.aaai.org": r"/index.php/[A-Za-z0-9_-]+/article/view/\d+(?:/\d+)?",
        "dl.acm.org": r"/doi/(?:abs/|pdf/|full/)?10\.\d{4,9}/[A-Za-z0-9._;()/:+-]+",
        "link.springer.com": r"/(?:article|chapter)/10\.\d{4,9}/[A-Za-z0-9._;()/:+-]+",
        "www.nature.com": r"/articles/[A-Za-z0-9._-]+(?:\.pdf)?",
        "www.science.org": r"/doi/(?:abs/|pdf/|full/)?10\.\d{4,9}/[A-Za-z0-9._;()/:+-]+",
        "proceedings.neurips.cc": r"/(?:paper_files/paper|paper)/\d{4}/(?:hash|file)/[A-Za-z0-9_-]+\.(?:html|pdf)",
    }
    if host in patterns:
        return re.fullmatch(patterns[host], path) is not None
    query = parse_qs(url.query)
    if host == "openreview.net" and path in {"/forum", "/pdf"}:
        return len(query.get("id", [])) == 1 and bool(re.fullmatch(r"[A-Za-z0-9_-]+", query["id"][0]))
    if host == "ieeexplore.ieee.org":
        return (re.fullmatch(r"/document/\d+/?", path) is not None
                or (path == "/stamp/stamp.jsp" and len(query.get("arnumber", [])) == 1
                    and query["arnumber"][0].isdigit()))
    return False


def eligible_primary_source(source: object) -> bool:
    """A location alone is not readable evidence; require the actual snapshot."""
    if not isinstance(source, dict):
        return False
    content, metadata = source.get("content"), source.get("metadata")
    return bool(
        source.get("kind") == "web" and isinstance(metadata, dict)
        and metadata.get("status") == "extracted"
        and isinstance(content, str) and content.strip()
        and hashlib.sha256(content.encode("utf-8")).hexdigest() == source.get("sha256")
        and approved_primary_url(source.get("uri"))
    )


def terminal_primary_candidates(sources: dict, personally_read: set) -> list[str]:
    """Permit one final extraction opportunity without extending model turns."""
    if any(eligible_primary_source(source) and key in personally_read
           for key, source in sources.items()):
        return []
    return [key for key, source in sources.items()
            if isinstance(source, dict) and source.get("kind") in {"search", "web"}
            and approved_primary_url(source.get("uri")) and key not in personally_read]
