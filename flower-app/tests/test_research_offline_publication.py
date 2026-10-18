"""Real saved evidence reaches the typed UI contract without any live research."""

import builtins
import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

from fusion_agent.research.engine import ResearchEngine, offline_report
from fusion_agent.research.evidence import EvidenceCatalog, resolve_pointer
from fusion_agent.research.grounded_facts import build_grounded_facts
from fusion_agent.research.providers import Settings
from fusion_agent.research.service import ResearchService

ROOT = Path(__file__).resolve().parents[2]
TRIAL = "12eddcebd2c548d889b41d3f2adace91"
RECORD = f"examples/evidence/shadow/{TRIAL}/comparison.json"


@pytest.fixture(params=[
    "dc0df0a81413479e979a18895e1fe2fe", "fbb83d8ee93a424497aeacc0a2633dc3", TRIAL,
], ids=["sac_actor_lr", "sac_mix", "td3_coordinates"])
def saved_source(request):
    source = EvidenceCatalog(ROOT).snapshot(f"examples/evidence/shadow/{request.param}/comparison.json")
    source.update(record_id=source["id"], id="E001")
    return source


def test_actual_dataset_offline_inspection_is_source_bound_and_never_a_model_draft(saved_source, monkeypatch):
    original = copy.deepcopy(saved_source)
    original_import = builtins.__import__

    def reject_live_import(name, *args, **kwargs):
        if name.startswith(("langgraph", "openai", "tavily")):
            pytest.fail("Offline saved inspection must not import live research dependencies")
        return original_import(name, *args, **kwargs)

    def reject_model_call(*args, **kwargs):
        pytest.fail("Offline saved inspection must not request model inference")

    monkeypatch.setattr(builtins, "__import__", reject_live_import)
    monkeypatch.setattr(ResearchEngine, "request", reject_model_call)
    report = offline_report([saved_source])
    publication = report["grounded_publication"]
    assert publication["schema"] == "fusion-grounded-publication-v1"
    assert publication["status"] == "insufficient_evidence"  # No invented literature or research completion.
    assert publication["literature_excerpts"] == []
    assert len(publication["facts"]) == 12
    trial = json.loads(saved_source["content"])["provenance"]["trial_id"]
    assert all(card["source_label"] == f"Saved shadow trial {trial}" for card in publication["facts"])
    assert report["answer"] == publication["answer"]
    assert report["claims"] == publication["facts"]
    assert report["limitations"] == publication["limitations"]
    assert "Offline saved evidence inspection" in report["answer"]
    assert "Current research-run" not in report["answer"]
    assert "model_interpretation" not in report and "judge" not in report and "model_draft" not in report
    assert report["evidence_sufficient"] is False
    assert report["inspection"] == {
        "mode": "offline", "model_calls": 0, "retrieval_calls": 0, "judge_calls": 0,
        "new_physics_episodes": 0, "scientific_research_complete": False, "source_ids": ["E001"],
    }
    assert report["orchestration"] == {
        "framework": "deterministic_inspection", "agents": [],
        "status": "inspection_complete", "full_workflow_reexecuted": False,
    }
    catalogue = {card["id"]: card for card in build_grounded_facts(saved_source)}
    for card in publication["facts"]:
        assert all(card[key] == value for key, value in catalogue[card["id"]].items())
        for citation in card["evidence"]:
            assert json.loads(citation["quote"]) == resolve_pointer(saved_source["content"], citation["locator"])
    for kind in ("paired_raw_endpoint", "paired_competence"):
        assert {card["context"]["arm"] for card in publication["facts"] if card["fact_type"] == kind} == {
            "local", "reset_only", "uniform", "clipped"}
    proposal = publication["recommendation"]
    recorded = json.loads(saved_source["content"])["next_run_proposal"]
    assert proposal["code"] == "keep_baseline" and proposal["request"] == recorded["request"]
    assert proposal["budget"]["total_episodes"] == 112
    assert proposal["automatic_apply"] is proposal["automatic_launch"] is False
    assert publication["scientific_efficacy_established"] is publication["automatic_policy_changes_allowed"] is False
    assert saved_source == original


def test_actual_case3_offline_service_and_export_preserve_inspection_complete_without_providers(tmp_path):
    folder = ROOT / RECORD.rsplit("/", 1)[0]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.rglob("*") if path.is_file()}

    def forbidden_factory(settings):
        pytest.fail("Offline inspection must not create a provider")

    service = ResearchService(
        ROOT, tmp_path / "research", settings_factory=Settings,
        model_factory=forbidden_factory, retrieval_factory=forbidden_factory,
    )
    run_id = service.start("Inspect the saved TD3 coordinate trial", [RECORD], "offline")["run"]["id"]
    deadline = time.monotonic() + 5
    while service.snapshot()["active"] and time.monotonic() < deadline:
        time.sleep(.005)
    snapshot = service.snapshot()
    assert snapshot["active"] is False
    run = snapshot["run"]
    assert run["status"] == "inspection_complete" and run["mode"] == "offline"
    assert run["allow_shadow"] is False
    assert run["provider"] == {"model": None, "endpoint": None}
    assert run["report"]["orchestration"]["status"] == "inspection_complete"
    assert len(run["report"]["grounded_publication"]["facts"]) == 12
    assert not any(event["kind"] in {"shadow_start", "model_usage", "search", "extract", "grounded_review"}
                   for event in run["events"])
    export = service.export(run_id)
    assert export["report"] == run["report"]
    assert export["sources"][0] == service.source(run_id, "E001")
    assert export["report"]["grounded_publication"]["recommendation"]["budget"]["total_episodes"] == 112
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.rglob("*") if path.is_file()}


def test_offline_saved_invalid_hash_is_not_promoted_to_typed_facts(saved_source):
    saved_source["content"] += " "
    with pytest.raises(ValueError, match="content hash"):
        offline_report([saved_source])


def test_offline_saved_missing_competence_remains_explicitly_incomplete(saved_source):
    document = json.loads(saved_source["content"])
    document["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0].pop("candidate_competent")
    saved_source["content"] = json.dumps(document)
    saved_source["sha256"] = hashlib.sha256(saved_source["content"].encode()).hexdigest()
    report = offline_report([saved_source])
    assert not report["verification"]["comparison_coverage"]["complete"]
    assert "Incomplete saved shadow comparison" in report["answer"]
    assert report["grounded_publication"]["recommendation"] is None
    assert report["evidence_sufficient"] is False


def test_offline_multiple_saved_sources_keep_legacy_inventory_without_pooled_publication(saved_source):
    other = copy.deepcopy(saved_source)
    other["id"] = "E002"
    report = offline_report([saved_source, other])
    assert "grounded_publication" not in report
    assert report["next_runs"] == []
    assert "Local inspection complete" in report["stop_reason"]
