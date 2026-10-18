"""Publication locations are conservative routing criteria, not truth labels."""

import copy
import hashlib

import pytest

from fusion_agent.research.primary_sources import (
    approved_primary_url,
    eligible_primary_source,
    terminal_primary_candidates,
)


@pytest.mark.parametrize("url", [
    "https://proceedings.mlr.press/v80/fujimoto18a.html",
    "https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf",
    "https://arxiv.org/abs/1802.09477", "https://arxiv.org/pdf/1802.09477v3.pdf",
    "https://arxiv.org/html/2607.22002v1",
    "https://openreview.net/forum?id=Abc_123", "https://openreview.net/pdf?id=Abc-123",
    "https://ojs.aaai.org/index.php/AAAI/article/view/34086/36241",
    "https://dl.acm.org/doi/pdf/10.5555/3504035.3504427",
    "https://ieeexplore.ieee.org/document/1234567",
    "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1234567",
    "https://link.springer.com/article/10.1007/s10994-019-05788-0",
    "https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab123-Abstract.html",
    "https://jmlr.org/papers/v25/23-001.html",
])
def test_recognized_paper_locations(url):
    assert approved_primary_url(url)


@pytest.mark.parametrize("url", [
    None, {}, "https://tinycto.tv/ml-atlas/literature/fujimoto-2018",
    "https://medium.com/paper", "https://docs.cleanrl.dev/rl-algorithms/td3",
    "https://proceedings.mlr.press.evil.test/v80/fujimoto18a.html",
    "https://proceedings.mlr.press@evil.test/v80/fujimoto18a.html",
    "https://evil.test@proceedings.mlr.press/v80/fujimoto18a.html",
    "https://proceedings.mlr.press/", "https://arxiv.org/search/?query=TD3",
    "https://arxiv.org/abs/fixture", "https://openreview.net/forum",
    "http://arxiv.org/abs/1802.09477", "https://arxiv.org:123/abs/1802.09477",
    "https://arxiv.org/abs/../1802.09477", "https://arxiv.org/abs/%31%38%30%32.09477",
    "https://arxiv.org/abs/1802.09477\n", "https://[badhost/abs/1802.09477",
])
def test_summaries_discovery_and_host_spoofing_are_ineligible(url):
    assert not approved_primary_url(url)


def source():
    text = "Captured primary article text, which still needs relevance review."
    return {"id": "W1", "uri": "https://arxiv.org/abs/1802.09477", "kind": "web",
            "content": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "metadata": {"status": "extracted"}}


@pytest.mark.parametrize("change", [
    {"kind": "search"}, {"metadata": {"status": "discovery_only"}},
    {"metadata": None}, {"sha256": "0" * 64}, {"content": "changed"},
])
def test_readable_hash_bound_snapshot_required(change):
    item = source()
    assert eligible_primary_source(item)
    item.update(change)
    assert not eligible_primary_source(item)


def test_terminal_extraction_only_when_no_eligible_primary_personally_read():
    item = source()
    secondary = copy.deepcopy(item)
    secondary.update(id="W2", uri="https://example.org/paper-summary")
    sources = {"W1": item, "W2": secondary}
    assert terminal_primary_candidates(sources, {"W2"}) == ["W1"]
    assert terminal_primary_candidates(sources, {"W1"}) == []
    item.update(kind="search", metadata={"status": "discovery_only"})
    assert terminal_primary_candidates(sources, set()) == ["W1"]
