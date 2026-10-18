"""Public retrieval must not receive paths from the local demo evidence tree."""

import pytest

from fusion_agent.research.retrieval_scope import (
    public_literature_text,
    require_public_query,
)


@pytest.mark.parametrize("path", [
    "examples/evidence/manifest.md", "examples/READMe.md", "examples/record",
    r"examples\evidence\notes.txt", "examples%2Fevidence%2Fnotes.md",
    "examples%252Fevidence%252Fnotes.md",
])
def test_public_examples_paths_are_removed_from_assignments_and_blocked_before_search(path):
    query = "Compare clipping using " + path
    assert public_literature_text(query, []) == "Compare clipping using [local artifact omitted]"
    with pytest.raises(ValueError, match="Public search query rejected"):
        require_public_query(query, [])


@pytest.mark.parametrize("query", [
    "federated reinforcement learning examples of clipped updates",
    "https://publisher.org/examples/paper",
    "Understanding Clipping for Federated Learning arXiv:2106.13673",
])
def test_public_literature_queries_are_not_changed(query):
    assert public_literature_text(query, []) == query
    require_public_query(query, [])
