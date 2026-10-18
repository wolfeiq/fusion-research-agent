"""Synthetic endpoint evidence only; no physics, learners or providers run."""

import hashlib
import json
from copy import deepcopy

import pytest

from fusion_agent.research.evidence import EvidenceCatalog, EvidenceError
from fusion_agent.research.shadow_analysis import (
    ARMS,
    RESULT_SCHEMA,
    analyze_shadow_result,
    attach_execution_status,
    canonical_sha256,
)


def summary(cost=1.0, competent=True, cases=2):
    return {
        "mean_cost": cost, "mean_beta_mae": 0.001,
        "mean_Te_profile_rmse_keV": 0.01, "competent": competent,
        "cases": cases, "complete_cases": cases,
        "cases_with_events": 0, "cases_with_invalid_physics": 0,
    }


@pytest.fixture
def worker_result():
    conditions = {"gate_factors": [1.0], "report_factors": [0.95, 1.05],
                  "training_shots": 2, "windows": 30}
    condition_hash = canonical_sha256(conditions)
    arms = {name: {"raw": summary(), "deployed": summary(),
                   "gate": {"accepted": True} if name in ("uniform", "clipped") else None}
            for name in ARMS}
    seed = {"seed": 17, "status": "complete", "conditions_sha256": condition_hash,
            "training_stable": True, "initial": summary(), "arms": arms,
            "rounds": [{"round": 1, "target_training_shots_per_arm": 2,
                        "arms": deepcopy(arms)}]}
    branch = {"status": "complete", "configs": {"td3": {"mix": 0.1}},
              "seeds": [seed], "accounting": {"training_episodes": 16,
                                               "evaluation_episodes": 12}}
    return {"schema": RESULT_SCHEMA, "status": "complete",
            "config": {"schema": "fusion-shadow-config-v1", "change": "td3_actor_lr",
                       "value": .00005, "seeds": [17], "rounds": 1},
            "conditions": conditions, "conditions_sha256": condition_hash,
            "baseline": deepcopy(branch), "candidate": deepcopy(branch),
            "failures": []}


def change_arm(result, arm, **changes):
    seed = result["candidate"]["seeds"][0]
    seed["arms"][arm]["raw"].update(changes)
    seed["rounds"][-1]["arms"] = deepcopy(seed["arms"])


def test_one_seed_lower_cost_is_only_repeat_recommendation(worker_result):
    change_arm(worker_result, "uniform", mean_cost=0.8)
    analysis = analyze_shadow_result(worker_result)
    pair = analysis["candidate_vs_baseline_raw_endpoint"]["uniform"]
    assert pair["paired_training_seeds"] == 1
    assert pair["lower_cost_seeds"] == 1
    assert pair["median_paired_cost_difference"] == pytest.approx(-0.2)
    assert analysis["recommendation"]["code"] == "repeat_varied_paired_seeds"
    assert analysis["recommendation"]["efficacy_claim"] is False
    assert analysis["recommendation"]["automatic_apply"] is False
    assert analysis["recommendation"]["automatic_launch"] is False
    assert "speedup" in " ".join(analysis["limitations"])
    assert analysis["worker_result_sha256"] == canonical_sha256(worker_result)


def test_same_endpoints_are_inconclusive_and_inputs_unchanged(worker_result):
    original = deepcopy(worker_result)
    analysis = analyze_shadow_result(worker_result)
    assert analysis["recommendation"]["code"] == "inconclusive"
    assert worker_result == original


@pytest.mark.parametrize("cost", [.8, 1.0, 1.2])
def test_next_run_repeats_same_hypothesis_even_for_null_and_negative(worker_result, cost):
    change_arm(worker_result, "uniform", mean_cost=cost)
    proposal = analyze_shadow_result(worker_result)["next_run_proposal"]
    assert proposal["request"] == {"schema": "fusion-shadow-config-v1", "change": "td3_actor_lr",
                                   "value": .00005, "seeds": [29, 43], "rounds": 1}
    assert proposal["budget"]["total_episodes"] == 112
    assert proposal["budget"]["training_episodes"] == 64
    assert proposal["budget"]["evaluation_episodes"] == 48
    assert proposal["budget"]["maximum_windows"] == 3360
    assert proposal["automatic_launch"] is False
    assert "Unused by this saved trial only" in proposal["seed_selection_scope"]


def test_next_seed_selection_excludes_all_current_trial_seeds(worker_result):
    worker_result["config"]["seeds"].append(29)
    for branch in ("baseline", "candidate"):
        row = deepcopy(worker_result[branch]["seeds"][0])
        row["seed"] = 29
        worker_result[branch]["seeds"].append(row)
    assert analyze_shadow_result(worker_result)["next_run_proposal"]["request"]["seeds"] == [43, 59]


def test_worse_raw_cannot_be_hidden_by_deployed_fallback(worker_result):
    change_arm(worker_result, "clipped", mean_cost=1.2)
    row = analyze_shadow_result(worker_result)["candidate_vs_baseline_raw_endpoint"]["clipped"]["by_seed"][0]
    assert row["mean_cost_difference"] == pytest.approx(0.2)
    assert row["deployed_comparison"]["mean_cost_difference"] == 0
    assert analyze_shadow_result(worker_result)["recommendation"]["code"] == "keep_baseline"


def test_competence_loss_blocks_adoption_even_with_lower_cost(worker_result):
    change_arm(worker_result, "uniform", mean_cost=0.8, competent=False,
               mean_Te_profile_rmse_keV=0.2)
    analysis = analyze_shadow_result(worker_result)
    assert analysis["recommendation"]["code"] == "keep_baseline"
    row = analysis["candidate_vs_baseline_raw_endpoint"]["uniform"]["by_seed"][0]
    assert row["mean_beta_mae_difference"] == 0
    assert row["mean_Te_profile_rmse_keV_difference"] == pytest.approx(0.19)


@pytest.mark.parametrize("arm,metric,value", [
    ("local", "mean_cost", 1.1),
    ("reset_only", "mean_cost", 1.1),
    ("uniform", "mean_beta_mae", .002),
    ("clipped", "mean_Te_profile_rmse_keV", .011),
])
def test_all_arms_and_separate_metrics_count_in_regression_guard(worker_result, arm, metric, value):
    change_arm(worker_result, "uniform", mean_cost=.8)
    change_arm(worker_result, arm, **{metric: value})
    assert analyze_shadow_result(worker_result)["recommendation"]["code"] == "keep_baseline"


@pytest.mark.parametrize("field,value", [
    ("training_stable", False), ("training_stable", None),
    ("status", "failed"),
])
def test_failed_or_unverified_learning_blocks_recommendation(worker_result, field, value):
    worker_result["candidate"]["seeds"][0][field] = value
    change_arm(worker_result, "uniform", mean_cost=0.8)
    assert analyze_shadow_result(worker_result)["recommendation"]["code"] == "diagnose_before_comparison"
    assert analyze_shadow_result(worker_result)["next_run_proposal"]["request"] is None


def test_missing_seed_is_preserved_not_dropped(worker_result):
    worker_result["config"]["seeds"].append(29)
    analysis = analyze_shadow_result(worker_result)
    assert analysis["status"] == "partial"
    assert len(analysis["failures"]) == 2
    assert analysis["recommendation"]["code"] == "diagnose_before_comparison"
    assert analysis["expected_training_seeds"] == [17, 29]


def test_mismatched_conditions_cannot_form_pair(worker_result):
    worker_result["candidate"]["seeds"][0]["conditions_sha256"] = "a" * 64
    analysis = analyze_shadow_result(worker_result)
    assert analysis["fairness"]["same_recorded_conditions"] is False
    assert analysis["candidate_vs_baseline_raw_endpoint"]["uniform"]["paired_training_seeds"] == 0
    assert analysis["recommendation"]["code"] == "repair_comparison_contract"


def test_invalid_physics_preserved_and_not_averaged_away(worker_result):
    change_arm(worker_result, "uniform", cases_with_invalid_physics=1,
               competent=False, mean_beta_mae=None, mean_Te_profile_rmse_keV=None)
    analysis = analyze_shadow_result(worker_result)
    assert analysis["candidate_vs_baseline_raw_endpoint"]["uniform"]["paired_training_seeds"] == 0
    assert analysis["branches"]["candidate"]["seeds"][17]["arms"]["uniform"]["raw"]["cases_with_invalid_physics"] == 1
    assert analysis["recommendation"]["code"] == "diagnose_before_comparison"


def test_case_count_is_not_independent_seed_count(worker_result):
    for branch in ("baseline", "candidate"):
        seed = worker_result[branch]["seeds"][0]
        for arm in seed["arms"].values():
            for key in ("raw", "deployed"):
                arm[key].update(cases=10, complete_cases=10)
        seed["rounds"][-1]["arms"] = deepcopy(seed["arms"])
    result = analyze_shadow_result(worker_result)
    assert result["candidate_vs_baseline_raw_endpoint"]["uniform"]["paired_training_seeds"] == 1


def test_mismatched_endpoint_case_counts_not_comparable(worker_result):
    change_arm(worker_result, "uniform", cases=1, complete_cases=1)
    result = analyze_shadow_result(worker_result)
    assert result["recommendation"]["code"] == "repair_comparison_contract"


def test_mixed_seeds_preserve_losses_and_median(worker_result):
    worker_result["config"]["seeds"].append(29)
    for branch in ("baseline", "candidate"):
        seed = deepcopy(worker_result[branch]["seeds"][0])
        seed["seed"] = 29
        worker_result[branch]["seeds"].append(seed)
    change_arm(worker_result, "uniform", mean_cost=0.8)
    second = worker_result["candidate"]["seeds"][1]
    second["arms"]["uniform"]["raw"]["mean_cost"] = 1.4
    second["rounds"][-1]["arms"] = deepcopy(second["arms"])
    analysis = analyze_shadow_result(worker_result)
    stats = analysis["candidate_vs_baseline_raw_endpoint"]["uniform"]
    assert stats["lower_cost_seeds"] == stats["higher_cost_seeds"] == 1
    assert stats["median_paired_cost_difference"] == pytest.approx(0.1)
    assert analysis["recommendation"]["code"] == "keep_baseline"


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(schema="made-up"), "schema"),
    (lambda r: r["conditions"].update(windows=60), "hash"),
    (lambda r: r["config"].update(seeds=[17, 17]), "unique"),
    (lambda r: r["config"].update(seeds=[True]), "unique"),
    (lambda r: r["candidate"]["seeds"].append(deepcopy(r["candidate"]["seeds"][0])), "duplicate"),
    (lambda r: r["candidate"]["seeds"][0]["arms"].pop("local"), "four arms"),
    (lambda r: r["candidate"]["seeds"][0]["arms"]["uniform"]["raw"].update(mean_cost=float("nan")), "JSON"),
    (lambda r: r["candidate"]["seeds"][0]["arms"]["uniform"]["raw"].update(mean_cost=True), "numeric"),
    (lambda r: r["candidate"]["seeds"][0]["arms"]["uniform"]["raw"].update(cases_with_events=1), "competence"),
    (lambda r: r["candidate"]["seeds"][0]["arms"]["uniform"]["raw"].update(complete_cases=9), "inconsistent"),
    (lambda r: r["candidate"]["seeds"][0]["rounds"][0].update(target_training_shots_per_arm=0), "increase"),
    (lambda r: r["candidate"]["seeds"][0]["rounds"][0]["arms"]["uniform"]["raw"].update(mean_cost=1.1), "disagree"),
    (lambda r: r["config"].update(rounds=2), "configured rounds"),
])
def test_rejects_malformed_or_inconsistent_results(worker_result, mutation, match):
    mutation(worker_result)
    with pytest.raises(ValueError, match=match):
        analyze_shadow_result(worker_result)


def test_zero_control_cost_has_no_infinite_percentage(worker_result):
    for branch in ("baseline", "candidate"):
        seed = worker_result[branch]["seeds"][0]
        seed["arms"]["uniform"]["raw"]["mean_cost"] = 0
        seed["rounds"][-1]["arms"] = deepcopy(seed["arms"])
    analysis = analyze_shadow_result(worker_result)
    assert analysis["candidate_vs_baseline_raw_endpoint"]["uniform"]["by_seed"][0]["relative_cost_change"] is None


def test_failure_details_and_cost_accounting_retained(worker_result):
    worker_result.update(status="failed", failures=[{"stage": "training", "error": "nonfinite loss"}])
    analysis = analyze_shadow_result(worker_result)
    assert analysis["status"] == "failed"
    assert analysis["failures"] == worker_result["failures"]
    assert analysis["branches"]["baseline"]["accounting"]["training_episodes"] == 16
    assert analysis["branches"]["baseline"]["seeds"][17]["initial"]["competent"] is True


def test_killed_running_worker_keeps_partial_arms_and_pending_branch(worker_result):
    worker_result["status"] = "running"
    worker_result["baseline"]["status"] = "running"
    seed = worker_result["baseline"]["seeds"][0]
    seed["status"] = "running"
    seed["arms"] = {"local": seed["arms"]["local"]}
    seed["rounds"][0]["arms"] = deepcopy(seed["arms"])
    worker_result["candidate"].update(status="pending", seeds=[])
    result = analyze_shadow_result(worker_result)
    assert result["status"] == "partial"
    assert result["worker_reported_status"] == "running"
    assert list(result["branches"]["baseline"]["seeds"][17]["arms"]) == ["local"]
    assert result["failures"] == [{"branch": "candidate", "seed": 17, "status": "missing"}]
    assert result["recommendation"]["code"] == "diagnose_before_comparison"


def test_per_case_identity_mismatch_cannot_be_paired(worker_result):
    for branch in ("baseline", "candidate"):
        seed = worker_result[branch]["seeds"][0]
        for arm in seed["arms"].values():
            for kind in ("raw", "deployed"):
                arm[kind]["per_case"] = [
                    {"seed": 1, "scenario": {"factor": .95}},
                    {"seed": 2, "scenario": {"factor": 1.05}},
                ]
        seed["rounds"][-1]["arms"] = deepcopy(seed["arms"])
    candidate = worker_result["candidate"]["seeds"][0]
    candidate["arms"]["uniform"]["raw"]["per_case"][0]["seed"] = 3
    candidate["rounds"][-1]["arms"] = deepcopy(candidate["arms"])
    result = analyze_shadow_result(worker_result)
    assert result["candidate_vs_baseline_raw_endpoint"]["uniform"]["paired_training_seeds"] == 0
    assert result["recommendation"]["code"] == "repair_comparison_contract"


def test_optimizer_ledgers_are_summed_and_projection_disclosed(worker_result):
    for branch in ("baseline", "candidate"):
        worker_result[branch]["accounting"].update({
            "contribution_optimization": [
                {"arm": "uniform", "role": "heating_td3", "seed": 17,
                 "round": 1, "source": source, "trunk_sgd_updates": 25}
                for source in ("donor_cool", "donor_hot")
            ],
            "repair_optimization": [{"seed": 17, "round": 1, "arm": "reset_only",
                                     "roles": {"heating_td3": {"actor_adaptation_updates": 20}}}],
        })
    result = analyze_shadow_result(worker_result)
    account = result["branches"]["candidate"]["accounting"]
    assert account["contribution_optimization"]["summed_by_role"]["heating_td3"]["trunk_sgd_updates"] == 50
    assert account["contribution_optimization"]["record_count"] == 2
    assert "original per-record ledger" in account["repair_optimization"]["projection"]
    assert "no separate actual" in account["source_target_split"]


@pytest.mark.parametrize("execution_status", ["failed", "timed_out", "interrupted"])
def test_process_failure_overrides_positive_recommendation_without_erasing_data(worker_result, execution_status):
    change_arm(worker_result, "uniform", mean_cost=.8)
    comparison = analyze_shadow_result(worker_result)
    final = attach_execution_status(comparison, execution_status)
    assert final["recommendation"]["code"] == "diagnose_before_comparison"
    assert final["worker_reported_status"] == "complete"
    assert final["candidate_vs_baseline_raw_endpoint"] == comparison["candidate_vs_baseline_raw_endpoint"]
    assert final["status"] != "complete"
    assert final["next_run_proposal"]["request"] is None
    assert comparison["recommendation"]["code"] == "repeat_varied_paired_seeds"


def test_completed_execution_requires_complete_comparison(worker_result):
    worker_result["config"]["seeds"].append(29)
    with pytest.raises(ValueError, match="complete comparison"):
        attach_execution_status(analyze_shadow_result(worker_result), "completed")


def parent_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


@pytest.fixture
def saved_shadow(tmp_path, worker_result):
    identifier = "a" * 32
    relative = f"flower-app/.dashboard/shadow/{identifier}/comparison.json"
    folder = tmp_path / relative.rsplit("/", 1)[0]
    (folder / "output").mkdir(parents=True)
    archive = b"synthetic source archive bytes; not executable"
    config = worker_result["config"]
    manifest = {
        "schema": "fusion-shadow-manifest-v1", "id": identifier,
        "config": config, "config_sha256": hashlib.sha256(parent_bytes(config)).hexdigest(),
        "source_zip_sha256": hashlib.sha256(archive).hexdigest(),
        "source_hashes": {"hfmarl/synthetic.py": "0" * 64},
    }
    manifest["identity"] = hashlib.sha256(parent_bytes(manifest)).hexdigest()
    result_bytes = parent_bytes(worker_result)
    comparison = attach_execution_status(analyze_shadow_result(worker_result), "completed")
    comparison["provenance"] = {
        "trial_id": identifier, "manifest_identity": manifest["identity"],
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "source_zip_sha256": manifest["source_zip_sha256"],
    }
    run = {"id": identifier, "status": "completed", "config": config,
           "manifest": manifest, "comparison": comparison}
    for name, value in (("config.json", config), ("manifest.json", manifest),
                        ("comparison.json", comparison), ("run.json", run)):
        (folder / name).write_bytes(parent_bytes(value))
    (folder / "output/result.json").write_bytes(result_bytes)
    (folder / "source.zip").write_bytes(archive)
    return tmp_path, folder, relative


def test_shadow_catalog_is_exact_hash_bound_read_only_evidence(saved_shadow):
    root, folder, relative = saved_shadow
    snapshot = EvidenceCatalog(root).snapshot(relative)
    assert snapshot["metadata"]["adapter"] == "shadow_comparison"
    assert snapshot["metadata"]["status"] == "complete"
    assert hashlib.sha256(snapshot["content"].encode()).hexdigest() == snapshot["sha256"]
    assert hashlib.sha256((folder / "comparison.json").read_bytes()).hexdigest() == snapshot["metadata"]["original_sha256"]
    assert EvidenceCatalog(root).list_records()[0]["id"] == relative
    assert EvidenceCatalog(root).list_records()[0]["adapter"] == "shadow_comparison"
    with pytest.raises(EvidenceError, match="allowlist"):
        EvidenceCatalog(root).snapshot(relative.replace("comparison.json", "output/result.json"))


def recorded_clipping_channel():
    return {
        "source_ids": ["donor", "target"],
        "source_norms": {"donor": {"q0": 2.0, "q1": 4.0}, "target": {"q0": .5, "q1": .25}},
        "thresholds": {"q0": 1.0, "q1": 1.0},
        "clipping_factors": {"donor": {"q0": .5, "q1": .25}, "target": {"q0": 1.0, "q1": 1.0}},
        "clipped_sources": {"q0": ["donor"], "q1": ["donor"]},
        "proposal_hashes": {"uniform": "a" * 64, "clipped": "b" * 64},
    }


def add_recorded_clipping(result):
    for branch in ("baseline", "candidate"):
        result[branch]["seeds"][0]["rounds"][0]["aggregation"] = {
            "clipped": {role: recorded_clipping_channel() for role in ("heating_td3", "deposition_sac")},
            "uniform": {"must_not_expose_hypothetical_clipping": True},
        }


def test_catalog_exposes_verified_clipping_without_rewriting_history(worker_result, request):
    add_recorded_clipping(worker_result)
    root, folder, relative = request.getfixturevalue("saved_shadow")
    before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    snapshot = EvidenceCatalog(root).snapshot(relative)
    projected = json.loads(snapshot["content"])
    diagnostic = projected.pop("round_aggregation_diagnostics")
    assert diagnostic["status"] == "available"
    assert diagnostic["raw_result_sha256"] == hashlib.sha256(before[folder / "output/result.json"]).hexdigest()
    assert len(diagnostic["records"]) == 4
    row = diagnostic["records"][-1]
    assert row["configuration"] == "candidate" and row["seed"] == 17 and row["round"] == 1
    assert row["role"] == "deposition_sac" and row["clipped_source_counts"] == {"q0": 1, "q1": 1}
    assert row["clipping_factors"]["donor"] == {"q0": .5, "q1": .25}
    assert row["recorded_uniform_and_clipped_hashes_equal"] is False
    assert row["raw_json_pointer"] == "/candidate/seeds/0/rounds/0/aggregation/clipped/deposition_sac"
    assert "must_not_expose" not in json.dumps(diagnostic)
    assert {path: path.read_bytes() for path in before} == before
    assert "round_aggregation_diagnostics" not in json.loads(before[folder / "comparison.json"])
    assert json.loads(EvidenceCatalog(root).snapshot(relative)["content"]) == json.loads(snapshot["content"])


def test_clipping_diagnostics_preserve_branch_difference_and_bound_extra_fields(worker_result):
    from fusion_agent.research.shadow_telemetry import clipping_diagnostic_view

    add_recorded_clipping(worker_result)
    channel = worker_result["baseline"]["seeds"][0]["rounds"][0]["aggregation"]["clipped"]["heating_td3"]
    channel["thresholds"] = {"q0": 10.0, "q1": 10.0, "unbounded_extra": "x" * 10000}
    channel["clipping_factors"] = {name: {"q0": 1.0, "q1": 1.0} for name in channel["source_ids"]}
    channel["clipped_sources"] = {"q0": [], "q1": []}
    channel["proposal_hashes"]["clipped"] = channel["proposal_hashes"]["uniform"]
    original = deepcopy(worker_result)
    view = clipping_diagnostic_view(worker_result, "c" * 64)
    assert view["records"][0]["clipped_source_counts"] == {"q0": 0, "q1": 0}
    assert view["records"][0]["recorded_uniform_and_clipped_hashes_equal"] is True
    assert view["records"][2]["clipped_source_counts"] == {"q0": 1, "q1": 1}
    assert "unbounded_extra" not in json.dumps(view)
    assert worker_result == original


@pytest.mark.parametrize("mutation", [
    lambda channel: channel["clipping_factors"]["donor"].update(q0=1.0),
    lambda channel: channel["clipped_sources"].update(q0=[]),
    lambda channel: channel["source_norms"]["donor"].update(q0=float("nan")),
    lambda channel: channel.update(proposal_hashes=None),
])
def test_inconsistent_clipping_diagnostics_never_claim_activation(worker_result, mutation):
    from fusion_agent.research.shadow_telemetry import clipping_diagnostic_view

    add_recorded_clipping(worker_result)
    channel = worker_result["candidate"]["seeds"][0]["rounds"][0]["aggregation"]["clipped"]["deposition_sac"]
    mutation(channel)
    view = clipping_diagnostic_view(worker_result, "c" * 64)
    assert view["status"] == "partial_or_unavailable"
    assert view["records"][-1]["status"] == "invalid_recorded_telemetry"
    assert "clipped_source_counts" not in view["records"][-1]


@pytest.mark.parametrize("aggregation", [None, [], {"clipped": None}])
def test_absent_clipping_diagnostics_are_unknown_not_zero(worker_result, aggregation):
    from fusion_agent.research.shadow_telemetry import clipping_diagnostic_view

    worker_result["candidate"]["seeds"][0]["rounds"][0]["aggregation"] = aggregation
    view = clipping_diagnostic_view(worker_result, "c" * 64)
    assert view["status"] == "partial_or_unavailable" and view["records"] == []
    assert len(view["missing_json_pointers"]) == 4


@pytest.mark.parametrize("name", ["source.zip", "output/result.json", "config.json"])
def test_shadow_catalog_rejects_changed_hash_inputs(saved_shadow, name):
    root, folder, relative = saved_shadow
    path = folder / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(EvidenceError, match="hash|configuration|provenance"):
        EvidenceCatalog(root).snapshot(relative)


def test_shadow_catalog_rejects_forged_recommendation(saved_shadow):
    root, folder, relative = saved_shadow
    comparison = json.loads((folder / "comparison.json").read_text())
    comparison["recommendation"]["efficacy_claim"] = True
    (folder / "comparison.json").write_bytes(parent_bytes(comparison))
    with pytest.raises(EvidenceError, match="deterministic analysis"):
        EvidenceCatalog(root).snapshot(relative)


def test_shadow_catalog_rejects_manifest_identity_change(saved_shadow):
    root, folder, relative = saved_shadow
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["id"] = "b" * 32
    (folder / "manifest.json").write_bytes(parent_bytes(manifest))
    with pytest.raises(EvidenceError, match="identity"):
        EvidenceCatalog(root).snapshot(relative)


def test_shadow_catalog_rejects_nonterminal_run(saved_shadow):
    root, folder, relative = saved_shadow
    run = json.loads((folder / "run.json").read_text())
    run["status"] = "running"
    (folder / "run.json").write_bytes(parent_bytes(run))
    with pytest.raises(EvidenceError, match="terminal"):
        EvidenceCatalog(root).snapshot(relative)


def test_shadow_catalog_exposes_failed_process_as_failed(saved_shadow):
    root, folder, relative = saved_shadow
    comparison = json.loads((folder / "comparison.json").read_text())
    comparison = attach_execution_status(comparison, "failed")
    run = json.loads((folder / "run.json").read_text())
    run.update(status="failed", comparison=comparison)
    (folder / "comparison.json").write_bytes(parent_bytes(comparison))
    (folder / "run.json").write_bytes(parent_bytes(run))
    snapshot = EvidenceCatalog(root).snapshot(relative)
    assert snapshot["metadata"]["status"] == "failed"
    assert json.loads(snapshot["content"])["recommendation"]["code"] == "diagnose_before_comparison"


def test_shadow_catalog_missing_auxiliary_is_invalid_not_complete(saved_shadow):
    root, folder, relative = saved_shadow
    (folder / "source.zip").unlink()
    records = EvidenceCatalog(root).list_records()
    assert records[0]["status"] == "invalid"
    with pytest.raises(EvidenceError, match="missing"):
        EvidenceCatalog(root).snapshot(relative)


@pytest.mark.parametrize("relative", [
    "flower-app/.dashboard/shadow/../comparison.json",
    "flower-app/.dashboard/shadow/abc/comparison.json",
    "flower-app/.dashboard/shadow/" + "a" * 32 + "/source.zip",
    "flower-app/.dashboard/shadow/" + "a" * 32 + "/comparison.json/../config.json",
])
def test_shadow_allowlist_has_no_arbitrary_paths(tmp_path, relative):
    with pytest.raises(EvidenceError, match="allowlist"):
        EvidenceCatalog(tmp_path).snapshot(relative)


def test_shadow_catalog_detects_cross_file_change(saved_shadow, monkeypatch):
    root, folder, relative = saved_shadow
    catalog = EvidenceCatalog(root)
    original = catalog._read_stable
    count = 0

    def changed(path, **kwargs):
        nonlocal count
        data = original(path, **kwargs)
        count += 1
        if count == 6:
            comparison = folder / "comparison.json"
            comparison.write_bytes(comparison.read_bytes() + b" ")
        return data

    monkeypatch.setattr(catalog, "_read_stable", changed)
    with pytest.raises(EvidenceError, match="changed during"):
        catalog.snapshot(relative)
