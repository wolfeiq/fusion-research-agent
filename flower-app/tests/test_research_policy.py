"""Recorded norm arithmetic and proposal contracts; no providers or RL execute."""

import hashlib
import json
from copy import deepcopy

import pytest

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.policy import (
    SCHEMA,
    compare_clipping,
    dossier,
    draft_policy_trial,
)


def capture(value):
    content = json.dumps(value, sort_keys=True)
    return {
        "id": "E001",
        "kind": "experiment",
        "title": "Synthetic captured coordinator dossier",
        "uri": "federation:synthetic",
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "metadata": {"adapter": "federation_dossier"},
    }


@pytest.fixture
def recorded():
    return {
        "schema": SCHEMA,
        "manifest": {
            "identity": "frozen-manifest-17",
            "config": {
                "clipping_initial_norm_fraction": 0.1,
                "heating_critic_coordinates": "physical",
                "seeds": [17, 29],
                "rounds": 2,
                "episodes_per_round": 4,
                "donor_pretrain_episodes": 6,
                "facility_factors": [0.9, 1.0, 1.1],
                "gate_factors": [0.95, 1.05],
                "report_factors": [0.9, 1.0, 1.1],
                "beta_mae_tolerance": 0.005,
                "beta_coverage": 0.9,
                "profile_rmse_tolerance_keV": 0.02,
            },
            "learner_configs": {
                "td3": {"mix": 0.1, "actor_lr": 0.0003},
                "sac": {"mix": 0.2, "actor_lr": 0.0002, "alpha_lr": 0.0003},
            },
            "expected_budget": {"episodes": 42, "donor": 12, "target": 16},
        },
        "jobs": [
            {
                "id": "clipped-seed17",
                "rounds": [
                    {
                        "round": 1,
                        "channels": {
                            "heating_td3": {
                                "weights": {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
                                "thresholds": {"q0": 4, "q1": 2},
                                "source_norms": {
                                    "a": {"q0": 0, "q1": 3},
                                    "b": {"q0": 4, "q1": 6},
                                    "c": {"q0": 10, "q1": 0},
                                },
                                "proposal_hashes": {
                                    "uniform": "heat-u",
                                    "clipped": "heat-c",
                                },
                            },
                            "deposition_sac": {
                                "weights": {"a": 0.5, "b": 0.5},
                                "thresholds": {"q0": 4, "q1": 3},
                                "source_norms": {
                                    "a": {"q0": 2, "q1": 1},
                                    "b": {"q0": 4, "q1": 3},
                                },
                                "proposal_hashes": {
                                    "uniform": "beam",
                                    "clipped": "beam",
                                },
                            },
                        },
                    }
                ],
            }
        ],
    }


def heating(recorded):
    return recorded["jobs"][0]["rounds"][0]["channels"]["heating_td3"]


def inspect(source, scale=0.5, **kwargs):
    return compare_clipping(
        source,
        kwargs.get("job_id", "clipped-seed17"),
        kwargs.get("round_number", 1),
        scale,
    )


def test_clipping_exact_norm_caps_counts_and_triangle_bounds(recorded):
    source = capture(recorded)
    before = deepcopy(source)
    result = inspect(source)
    td3 = result["roles"]["heating_td3"]
    q0, q1 = td3["branches"]["q0"], td3["branches"]["q1"]
    assert q0["recorded_threshold"] == 4
    assert q0["candidate_threshold"] == 2
    assert q0["sources"]["a"]["recorded_factor"] == 1  # Zero norm.
    assert q0["sources"]["a"]["candidate_factor"] == 1
    assert q0["sources"]["b"]["recorded_factor"] == 1  # Exactly at cap.
    assert q0["sources"]["b"]["candidate_factor"] == 0.5
    assert q0["sources"]["c"]["recorded_factor"] == 0.4
    assert q0["sources"]["c"]["candidate_factor"] == 0.2
    assert q0["recorded_clipped_count"] == 1
    assert q0["candidate_clipped_count"] == 2
    assert q0["uniform_mean_norm_upper_bound"] == pytest.approx(14 / 3)
    assert q0["candidate_mean_norm_upper_bound"] == pytest.approx(4 / 3)
    assert q1["candidate_threshold"] == 1
    assert q1["sources"]["a"]["candidate_factor"] == pytest.approx(1 / 3)
    assert q1["sources"]["b"]["candidate_factor"] == pytest.approx(1 / 6)
    assert q1["candidate_clipped_count"] == 2
    assert q1["candidate_mean_norm_upper_bound"] == pytest.approx(2 / 3)
    assert td3["recorded_uniform_equals_clipped"] is False
    assert result["roles"]["deposition_sac"]["recorded_uniform_equals_clipped"] is True
    assert result["source_id"] == source["id"]
    assert result["source_sha256"] == source["sha256"]
    assert result["manifest_identity"] == "frozen-manifest-17"
    assert "no tensor reaggregation" in result["scope"]
    assert "not realized aggregate norms" in result["limitations"][0]
    assert source == before


@pytest.mark.parametrize(
    "scale,cap,count,bound",
    [
        (0.5, 2, 2, 4 / 3),
        (1, 4, 1, 8 / 3),
        (2, 8, 1, 4),
    ],
)
def test_supported_threshold_scales_preserve_uniform_weights(
    recorded,
    scale,
    cap,
    count,
    bound,
):
    branch = inspect(capture(recorded), scale)["roles"]["heating_td3"]["branches"]["q0"]
    assert branch["candidate_threshold"] == cap
    assert branch["candidate_clipped_count"] == count
    assert branch["candidate_mean_norm_upper_bound"] == pytest.approx(bound)
    assert [s["weight"] for s in branch["sources"].values()] == [1 / 3] * 3


@pytest.mark.parametrize(
    "scale", [True, False, 0, -1, 0.75, "1", float("nan"), float("inf")]
)
def test_clipping_rejects_unfrozen_or_non_numeric_scale(recorded, scale):
    with pytest.raises(ValueError):
        inspect(capture(recorded), scale)


@pytest.mark.parametrize("round_number", [True, False, 0, -1, 1.0, "1", 2])
def test_clipping_requires_an_exact_recorded_integer_round(recorded, round_number):
    with pytest.raises(ValueError):
        inspect(capture(recorded), round_number=round_number)


def test_job_identity_and_duplicate_rounds_are_rejected(recorded):
    with pytest.raises(ValueError):
        inspect(capture(recorded), job_id="unknown")
    recorded["jobs"][0]["rounds"] *= 2
    with pytest.raises(ValueError):
        inspect(capture(recorded))
    recorded["jobs"] *= 2
    with pytest.raises(ValueError):
        inspect(capture(recorded))


@pytest.mark.parametrize(
    "weights",
    [
        {"a": 0.2, "b": 0.3, "c": 0.5},  # Not the recorded uniform-cohort policy.
        {"a": 0.2, "b": 0.2, "c": 0.2},
        {"a": -0.1, "b": 0.5, "c": 0.6},
        {"a": 1.1, "b": 0, "c": 0},
        {"a": True, "b": 0, "c": 0},
        {"a": float("nan"), "b": 0.5, "c": 0.5},
        {"a": float("inf"), "b": 0, "c": 0},
        {"a": 0.5, "b": 0.5},
    ],
)
def test_nonuniform_or_invalid_recorded_weights_fail_closed(recorded, weights):
    heating(recorded)["weights"] = weights
    with pytest.raises(ValueError):
        inspect(capture(recorded))


@pytest.mark.parametrize("bad", [True, False, -0.1, "2", float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["source_norms", "thresholds"])
def test_norms_and_thresholds_reject_invalid_numbers(recorded, bad, field):
    if field == "source_norms":
        heating(recorded)[field]["a"]["q0"] = bad
    else:
        heating(recorded)[field]["q0"] = bad
    with pytest.raises(ValueError):
        inspect(capture(recorded))


@pytest.mark.parametrize("pair", [{"q0": 1}, {"q0": 1, "q1": 2, "q2": 3}, None, [1, 2]])
def test_malformed_source_branch_schema_is_a_safe_error(recorded, pair):
    heating(recorded)["source_norms"]["a"] = pair
    with pytest.raises(ValueError):
        inspect(capture(recorded))


def test_uniform_round_missing_diagnostics_is_explicitly_unavailable(recorded):
    channels = recorded["jobs"][0]["rounds"][0]["channels"]
    channels["heating_td3"] = {"proposal_hashes": {"uniform": "only-aggregate"}}
    channels["deposition_sac"] = {}
    result = inspect(capture(recorded))
    for role in result["roles"].values():
        assert role["status"] == "unavailable"
        assert "cannot recover donor deltas" in role["reason"]
        assert "branches" not in role


def test_overflowing_counterfactual_threshold_is_rejected(recorded):
    heating(recorded)["thresholds"]["q0"] = 1e308
    with pytest.raises(ValueError):
        inspect(capture(recorded), 2)


def test_absent_proposal_hashes_are_unknown_not_assumed_equal(recorded):
    heating(recorded).pop("proposal_hashes")
    result = inspect(capture(recorded))
    assert result["roles"]["heating_td3"]["recorded_uniform_equals_clipped"] is None


@pytest.mark.parametrize(
    "mutation", ["bytes", "hash", "kind", "schema", "manifest", "jobs"]
)
def test_dossier_integrity_and_schema_binding(recorded, mutation):
    source = capture(recorded)
    if mutation == "bytes":
        source["content"] += " "
    elif mutation == "hash":
        source["sha256"] = "0" * 64
    elif mutation == "kind":
        source["kind"] = "protocol"
    else:
        recorded[mutation] = "wrong"
        source = capture(recorded)
    with pytest.raises(ValueError):
        dossier(source)


@pytest.mark.parametrize(
    "change,candidate,support,pointer",
    [
        (
            "heating_critic_coordinates",
            "normalized",
            "runner_cli_supported",
            "/config/heating_critic_coordinates",
        ),
        (
            "clipping_initial_norm_fraction",
            0.05,
            "programmatic_configuration_only",
            "/config/clipping_initial_norm_fraction",
        ),
        ("td3_mix", 0.25, "requires_runner_change", "/learner_configs/td3/mix"),
        ("sac_mix", 0.25, "requires_runner_change", "/learner_configs/sac/mix"),
        (
            "td3_actor_lr",
            0.0001,
            "requires_runner_change",
            "/learner_configs/td3/actor_lr",
        ),
        (
            "sac_actor_lr",
            0.0001,
            "requires_runner_change",
            "/learner_configs/sac/actor_lr",
        ),
        (
            "sac_alpha_lr",
            0.0001,
            "requires_runner_change",
            "/learner_configs/sac/alpha_lr",
        ),
    ],
)
def test_proposals_bind_one_change_preserve_controls_and_never_apply(
    recorded,
    change,
    candidate,
    support,
    pointer,
):
    source = capture(recorded)
    before = deepcopy(source)
    result = draft_policy_trial(source, change, candidate)
    assert result["implementation_status"] == support
    assert result["change"]["json_pointer"] == pointer
    assert result["change"]["candidate"] == candidate
    assert result["change"]["before_recorded"] is True
    assert result["source_sha256"] == source["sha256"]
    assert result["manifest_identity"] == recorded["manifest"]["identity"]
    assert result["status"] == "proposal_only"
    assert result["applied"] is False and result["executable"] is False
    assert set(result["controls"]) == {
        "local",
        "reset_only",
        "initial_base_pull",
        "raw_federated",
        "deployed",
        "joint_sac",
        "calibration_fixed_command",
    }
    assert len(result["controls"]) == 7
    assert result["frozen_fields"]["seeds"] == [17, 29]
    assert (
        result["budget"]["recorded_matrix"] == recorded["manifest"]["expected_budget"]
    )
    assert result["budget"]["candidate_budget_status"] == "not_recomputed"
    assert "whole baseline-and-candidate matrix" in result["budget"]["requirement"]
    assert "preserve failures" in result["stop_rule"]
    assert any("confirmation" in item for item in result["prerequisites"])
    assert any("gate fallback" in item for item in result["prerequisites"])
    if support == "requires_runner_change":
        assert any(
            "configuration/schema hashes" in item for item in result["prerequisites"]
        )
    payload = {key: value for key, value in result.items() if key != "candidate_id"}
    assert (
        result["candidate_id"]
        == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    )
    assert source == before
    result["frozen_fields"]["seeds"].append(999)
    result["budget"]["recorded_matrix"]["episodes"] = -1
    assert source == before


@pytest.mark.parametrize(
    "change,value",
    [
        ("clipping_initial_norm_fraction", 0.001),
        ("clipping_initial_norm_fraction", 1),
        ("td3_mix", 0),
        ("td3_mix", 1),
        ("sac_actor_lr", 1e-6),
        ("sac_actor_lr", 1e-2),
    ],
)
def test_proposal_numeric_boundaries_are_inclusive(recorded, change, value):
    assert (
        draft_policy_trial(capture(recorded), change, value)["change"]["candidate"]
        == value
    )


@pytest.mark.parametrize(
    "change,value",
    [
        ("unknown", 0.1),
        ("heating_critic_coordinates", "standardized"),
        ("heating_critic_coordinates", True),
        ("clipping_initial_norm_fraction", 0),
        ("clipping_initial_norm_fraction", 1.01),
        ("td3_mix", -0.01),
        ("sac_mix", 1.01),
        ("td3_actor_lr", 1e-7),
        ("sac_alpha_lr", 0.1),
        ("td3_actor_lr", "0.001"),
        ("sac_mix", True),
        ("sac_actor_lr", float("nan")),
        ("sac_actor_lr", float("inf")),
        ("heating_critic_coordinates", "physical"),
        ("clipping_initial_norm_fraction", 0.1),
        ("td3_mix", 0.1),
    ],
)
def test_invalid_unsupported_or_unchanged_policy_candidates_rejected(
    recorded, change, value
):
    with pytest.raises(ValueError):
        draft_policy_trial(capture(recorded), change, value)


def test_absent_baseline_is_visible_and_requires_baseline_reconstruction(recorded):
    recorded["manifest"]["learner_configs"]["td3"].pop("mix")
    result = draft_policy_trial(capture(recorded), "td3_mix", 0.25)
    assert result["change"]["before"] is None
    assert result["change"]["before_recorded"] is False
    assert any("effective baseline" in item for item in result["prerequisites"])


def test_candidate_id_changes_with_captured_source_identity(recorded):
    first = draft_policy_trial(capture(recorded), "td3_mix", 0.25)
    recorded["manifest"]["identity"] = "other-frozen-manifest"
    second = draft_policy_trial(capture(recorded), "td3_mix", 0.25)
    assert first["candidate_id"] != second["candidate_id"]
    assert first["source_sha256"] != second["source_sha256"]


class NoProvider:
    def complete(self, *_args, **_kwargs):
        pytest.fail("Deterministic diagnostics must not call a model")

    def search(self, *_args, **_kwargs):
        pytest.fail("Deterministic diagnostics must not search the web")

    def extract(self, *_args, **_kwargs):
        pytest.fail("Deterministic diagnostics must not extract a web page")


def engine_for(source):
    events = []
    engine = ResearchEngine([source], NoProvider(), NoProvider(), events.append)
    return engine, events


def plan(engine):
    engine.execute(
        "plan_research",
        {
            "questions": ["What do saved rounds show?"],
            "reason": "Inspect recorded evidence before proposing a change.",
        },
    )


@pytest.mark.parametrize(
    "tool,arguments",
    [
        (
            "compare_clipping",
            {"job_id": "clipped-seed17", "round": 1, "threshold_scale": 0.5},
        ),
        ("draft_policy_trial", {"change": "td3_mix", "value": 0.25}),
    ],
)
@pytest.mark.parametrize("read_tool", ["inspect_federation", "read_experiment"])
def test_engine_requires_prior_read_and_registers_hash_bound_derived_sources(
    recorded,
    tool,
    arguments,
    read_tool,
):
    source = capture(recorded)
    before = deepcopy(source)
    engine, events = engine_for(source)
    call = {
        "source_id": "E001",
        "reason": "Inspect a bounded source-based candidate.",
        **arguments,
    }
    with pytest.raises(ValueError, match="plan_research"):
        engine.execute(tool, call)
    plan(engine)
    with pytest.raises(ValueError, match="before computing"):
        engine.execute(tool, call)
    assert len(engine.sources) == 1
    read = engine.execute(
        read_tool, {"source_id": "E001", "reason": "Read the captured round."}
    )
    assert read["sha256"] == source["sha256"]
    result = engine.execute(tool, call)
    assert result["id"].startswith("D")
    assert result["kind"] == "experiment"
    assert result["sha256"] == hashlib.sha256(result["content"].encode()).hexdigest()
    assert result["metadata"]["input_sha256"] == source["sha256"]
    assert result["metadata"]["input_source_id"] == "E001"
    assert result["metadata"]["tool"] == tool
    assert "not a new rollout" in result["metadata"]["scope"]
    assert engine.sources[result["id"]] == result
    assert result["id"] in engine.read_ids
    assert json.loads(result["content"])["source_sha256"] == source["sha256"]
    assert any(event.get("source") == result for event in events)
    again = engine.execute(tool, call)
    assert again["id"] == result["id"]
    assert len(engine.sources) == 2
    assert source == before


def test_engine_cannot_inspect_unselected_dossier_or_launch_training(recorded):
    engine, _ = engine_for(capture(recorded))
    plan(engine)
    with pytest.raises(ValueError):
        engine.execute(
            "inspect_federation",
            {"source_id": "unselected", "reason": "Try an unavailable source."},
        )
    with pytest.raises(ValueError, match="only read-only"):
        engine.execute("launch_training", {"reason": "Try launching a controller."})
    assert len(engine.sources) == 1


def test_engine_rechecks_snapshot_integrity_before_diagnostics(recorded):
    source = capture(recorded)
    engine, _ = engine_for(source)
    plan(engine)
    engine.execute(
        "inspect_federation", {"source_id": "E001", "reason": "Read the capture."}
    )
    source["content"] += " "
    with pytest.raises(ValueError, match="integrity"):
        engine.execute(
            "compare_clipping",
            {
                "source_id": "E001",
                "reason": "Read again.",
                "job_id": "clipped-seed17",
                "round": 1,
                "threshold_scale": 1,
            },
        )
    assert len(engine.sources) == 1
