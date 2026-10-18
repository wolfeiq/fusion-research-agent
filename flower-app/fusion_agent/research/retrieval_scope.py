"""Concrete local-artifact filters for public retrieval, not general-purpose DLP."""

from __future__ import annotations

import re
from urllib.parse import unquote

_LOCAL_MARKERS = re.compile(
    r"file://[^\s<>\"']+"
    r"|(?<!\w)[a-z]:[\\/][^\s<>\"']+"
    r"|\\\\[^\s<>\"']+"
    r"|(?<![\w:/])(?:\.{1,2}[\\/]|~[\\/]|\.(?:dashboard|codex)[\\/]"
    r"|(?:flower-app|results|docs|artifacts|checkpoints|hfmarl|scripts)[\\/])[^\s<>\"']+"
    r"|(?<![\w:/])/(?:Users|home|tmp|mnt|workspace|kaggle|private|var)/[^\s<>\"']+"
    r"|(?<![\w:])(?:[\w.-]+[\\/])+[\w.-]+\.(?:jsonl?|csv|tsv|npz|npy|pt|pth|ckpt|zip|log|ya?ml|toml)\b"
    r"|(?<![a-z0-9])[a-f0-9]{32,}(?![a-z0-9])"
    r"|(?<![a-z0-9])[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}(?![a-z0-9])",
    re.IGNORECASE,
)


def public_literature_text(value, sources):
    """Redact known local identifiers and path shapes from a planner assignment.

    Public question text and local catalog metadata are not passed to this role.
    This additional filter handles an assignment accidentally carrying an artifact
    reference. It does not classify all private facts or scientific measurements.
    """
    public = _LOCAL_MARKERS.sub("[local artifact omitted]", unquote(unquote(value)))
    for source in sources:
        if source.get("kind") not in ("experiment", "protocol"):
            continue
        for field in ("uri", "id"):
            marker = source.get(field)
            if isinstance(marker, str) and marker:
                marker = unquote(unquote(marker)).replace("\\", "/")
                pattern = re.escape(marker).replace("/", r"[\\/]")
                if field == "id":
                    pattern = r"(?<!\w)" + pattern + r"(?!\w)"
                public = re.sub(pattern, "[local artifact omitted]", public, flags=re.IGNORECASE)
    return public


def require_public_query(query, sources):
    """Reject concrete local-artifact references before a retrieval API call."""
    if public_literature_text(query, sources) != unquote(unquote(query)):
        raise ValueError(
            "Public search query rejected: local paths, selected record identifiers, or artifact IDs are not allowed. "
            "Search general scientific mechanisms or public paper titles/DOIs/arXiv IDs; local evidence is handled by the three local specialists."
        )

