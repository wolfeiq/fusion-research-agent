"""Shadow worker contract tests; the toy plant below is NOT TORAX evidence."""
import copy
import json
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_shadow_trial as worker

CONFIG = {"schema": "fusion-shadow-config-v1", "change": "td3_actor_lr", "value": .00005,
          "seeds": [17], "rounds": 1}


def test_validation_is_pure_and_imports_no_ml_runtime():
    script = "from scripts.run_shadow_trial import validate_config; import sys; assert 'torch' not in sys.modules; assert 'jax' not in sys.modules; assert 'torax' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


@pytest.mark.parametrize("flower_version", [None, "1.22.0"])
def test_runtime_metadata_records_optional_unused_flower(monkeypatch, flower_version):
    def versions(name):
        if name == "flwr" and flower_version is None:
            raise PackageNotFoundError(name)
        return flower_version if name == "flwr" else "test-required-version"

    monkeypatch.setattr(worker, "version", versions)
    metadata = worker.runtime_metadata()
    assert metadata["packages"]["flwr"] == flower_version
    assert metadata["packages"]["torax"] == "test-required-version"
    assert metadata["flower_transport_used"] is False


def test_runtime_metadata_does_not_hide_missing_numerical_dependency(monkeypatch):
    def versions(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(worker, "version", versions)
    with pytest.raises(PackageNotFoundError):
        worker.runtime_metadata()


@pytest.mark.parametrize("patch", [
    {"change": "arbitrary_python"}, {"change": []}, {"value": True},
    {"value": float("nan")}, {"value": float("inf")}, {"value": .002},
    {"value": 1e-8}, {"value": .0001}, {"seeds": [True]}, {"seeds": [1, 1]},
    {"seeds": []}, {"seeds": [1, 2, 3, 4]}, {"seeds": [-1]}, {"seeds": [1000001]},
    {"rounds": True}, {"rounds": 0}, {"rounds": 3}, {"command": "run something"},
    {"change": "heating_critic_coordinates", "value": "normalized"},
    {"change": "heating_critic_coordinates", "value": "invented"},
])
def test_rejects_unbounded_noop_and_executable_configuration(patch):
    with pytest.raises(ValueError):
        worker.validate_config({**CONFIG, **patch})


@pytest.mark.parametrize(("change", "value", "part", "field"), [
    ("td3_actor_lr", .00005, "td3", "actor_lr"),
    ("sac_actor_lr", .0001, "sac", "actor_lr"),
    ("sac_alpha_lr", .0001, "sac", "alpha_lr"),
    ("td3_mix", .1, "td3", "mix"), ("sac_mix", .1, "sac", "mix"),
    ("heating_critic_coordinates", "physical", "td3", "critic_action_coordinates"),
])
def test_candidate_changes_exactly_one_learner_setting(change, value, part, field):
    worker.validate_config({**CONFIG, "change": change, "value": value})
    baseline = dict(zip(("td3", "sac"), map(asdict, worker._learner_configs(worker.BASELINE))))
    candidate = dict(zip(("td3", "sac"), map(asdict, worker._learner_configs({**worker.BASELINE, change: value}))))
    baseline[part][field] = value
    assert baseline == candidate


def test_budget_counts_all_source_gate_and_reference_rollouts():
    first = worker.specification(CONFIG)["budget"]
    assert first["training_episodes"] == 32
    assert first["evaluation_episodes"] == 24
    assert first["total_episodes"] == 56
    second = worker.specification({**CONFIG, "seeds": [17, 29, 41], "rounds": 2})["budget"]
    assert second["total_episodes"] == 276
    assert second["maximum_windows"] == 8280
    assert second["training_episodes"] + second["evaluation_episodes"] == 276


class SyntheticPhysicalEnv:
    """Thirty cheap transitions for optimizer/protocol verification only."""
    task = SimpleNamespace(steps_per_shot=30)

    def __init__(self, factor):
        self.factor = factor
        self.core = self

    def reset(self):
        self.t = 0.
        self.index = 0
        self.x = np.zeros(37, np.float32)
        self.x[0] = self.factor
        return self.x.copy(), {"synthetic_test_only": True}

    def step(self, action):
        assert np.shape(action) == (4,) and np.isfinite(action).all()
        self.index += 1
        self.t = self.index * .5
        self.x[:4] += np.asarray(action, np.float32) * .001
        self.x[4] = self.index / 30
        reward = -float(np.square(np.asarray(action) - .2).sum()) * .01
        return self.x.copy(), reward, False, self.index == 30, {"violations": []}

    def read_scalars(self):
        return {"beta_N": .4 + float(self.x[0] - self.factor) * .01,
                "q95": 3., "fgw_n_e_line_avg": .4}

    def read_profiles(self):
        return {"rho_norm": [0., 1.], "T_e": [4. + float(self.x[2]), 3.]}


@pytest.fixture
def synthetic_setup(tmp_path, monkeypatch):
    reference = {"beta_N": .4, "profile_rho": [.1, .9], "target_Te_keV": [3.9, 3.1],
                 "joint_action": [0., 0., 0., 1.], "device": "synthetic_test_only"}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    environments = {}

    def factory(_reference):
        def environment(factor):
            if factor not in environments:
                environments[factor] = SyntheticPhysicalEnv(factor)
            return environments[factor]
        return environment, [1., 1.]

    monkeypatch.setattr(worker, "_make_environments", factory)
    return reference_path, tmp_path / "output"


@pytest.fixture(scope="module")
def completed_pilot(tmp_path_factory):
    """Real optimizers/aggregation, synthetic physics, two rounds, no Flower metadata."""
    directory = tmp_path_factory.mktemp("shadow-pilot")
    reference = {"beta_N": .4, "profile_rho": [.1, .9], "target_Te_keV": [3.9, 3.1],
                 "joint_action": [0., 0., 0., 1.], "device": "synthetic_test_only"}
    reference_path = directory / "reference.json"
    reference_path.write_text(json.dumps(reference))
    installed_version = worker.version

    def without_flower(name):
        if name == "flwr":
            raise PackageNotFoundError(name)
        return installed_version(name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker, "_make_environments", lambda _: (SyntheticPhysicalEnv, [1., 1.]))
        patch.setattr(worker, "version", without_flower)
        result = worker.run({**CONFIG, "rounds": 2}, reference_path, directory / "output")
    return result, directory / "output"


def test_pilot_really_updates_both_agents_and_accounts_for_work(completed_pilot):
    result, output = completed_pilot
    assert result["status"] == "complete" and not result["failures"]
    assert result["runtime"]["packages"]["flwr"] is None
    assert result["runtime"]["flower_transport_used"] is False
    assert result["conditions_sha256"] == worker._digest(result["conditions"])
    for variant in ("baseline", "candidate"):
        item = result[variant]
        assert item["status"] == "complete"
        assert item["accounting"]["training_episodes"] == 24
        assert item["accounting"]["evaluation_episodes"] == 22
        assert item["accounting"]["windows"] == 46 * 30
        assert item["accounting"]["training_optimizer_updates"]["heating"] > 0
        assert item["accounting"]["training_optimizer_updates"]["beam"] > 0
        assert len(item["accounting"]["contribution_optimization"]) == 24
        assert len(item["accounting"]["repair_optimization"]) == 6
        seed = item["seeds"][0]
        assert seed["status"] == "complete" and seed["training_stable"]
        assert seed["conditions_sha256"] == result["conditions_sha256"]
        for arm in worker.ARMS:
            assert seed["arms"][arm]["learner_updates"]["heating"] > 0
            assert seed["arms"][arm]["learner_updates"]["beam"] > 0
            assert seed["arms"][arm]["raw"]["optimizer_updates"] == {"heating": 0, "beam": 0, "joint": 0}
    assert len(list((output / "episodes").glob("*.json"))) == 92
    assert len(list((output / "aggregation").glob("*.npz"))) == 8
    assert json.loads((output / "progress.json").read_text())["completed_episodes"] == 92


def test_paired_conditions_initialization_and_raw_gate_separation(completed_pilot):
    result, _ = completed_pilot
    baseline, candidate = [result[v]["seeds"][0] for v in ("baseline", "candidate")]
    assert baseline["initial_representation_hashes"] == candidate["initial_representation_hashes"]
    assert baseline["initial"] == candidate["initial"]
    for variant in (baseline, candidate):
        for row in variant["rounds"]:
            pairs = [row["arms"][arm]["raw"]["per_case"][0] for arm in worker.ARMS]
            assert len({p["seed"] for p in pairs}) == 1
            assert all(p["scenario"] == {"initial_temperature_factor": 1.05} for p in pairs)
            assert row["arms"]["reset_only"]["gate"] is None
            assert "reset_only" not in row["aggregation"]
            for arm in ("uniform", "clipped"):
                item = row["arms"][arm]
                selected = item["raw"] if item["gate"]["accept"] else item["incumbent"]
                assert item["deployed"] == selected


def test_real_aggregation_common_bases_and_clipping_are_recomputable(completed_pilot):
    from hfmarl.federation.representation import model_hash
    result, output = completed_pilot
    for variant in ("baseline", "candidate"):
        rounds = result[variant]["seeds"][0]["rounds"]
        for arm in ("uniform", "clipped"):
            for role in ("heating_td3", "deposition_sac"):
                first, second = [row["aggregation"][arm][role] for row in rounds]
                assert second["base_hash"] == first["proposal_hashes"][arm]
                assert set(first["source_ids"]) == {"donor_cool", "donor_warm", "cold_target"}
                with np.load(output / "aggregation" / f"{variant}-17-{arm}-1.npz", allow_pickle=False) as archive:
                    prefix = f"{role}/base/"
                    base = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                    assert model_hash(base) == first["base_hash"]
                    for key, value in base.items():
                        branch = key.split(".")[0]
                        delta = sum(archive[f"{role}/delta/{source}/{key}"].astype(np.float64) *
                            (first["clipping_factors"][source][branch] if arm == "clipped" else 1.)
                            for source in first["source_ids"]) / 3
                        expected = (value.astype(np.float64) + delta).astype(value.dtype)
                        np.testing.assert_array_equal(archive[f"{role}/proposal/{key}"], expected)


def test_training_failure_is_preserved_without_continuing_transfer(synthetic_setup, monkeypatch):
    from hfmarl.experiments import coupled_control
    original = coupled_control.episode

    def failing(*args, **kwargs):
        if kwargs.get("train"):
            raise FloatingPointError("test optimizer failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(coupled_control, "episode", failing)
    reference, output = synthetic_setup
    with pytest.raises(FloatingPointError, match="test optimizer"):
        worker.run(CONFIG, reference, output)
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "partial"
    assert result["baseline"]["seeds"][0]["status"] == "failed"
    assert not result["baseline"]["seeds"][0]["training_stable"]
    assert result["candidate"]["status"] == "pending"
    assert not list((output / "aggregation").glob("*"))
    assert result["failures"][0]["type"] == "FloatingPointError"
    assert any(json.loads(path.read_text())["status"] == "started" for path in (output / "episodes").glob("*"))


def test_nonempty_outputs_and_short_deadlines_do_not_overwrite_evidence(synthetic_setup):
    reference, output = synthetic_setup
    with pytest.raises(TimeoutError, match="wall-time"):
        worker.run(CONFIG, reference, output, max_seconds=.000001)
    first = copy.deepcopy(json.loads((output / "result.json").read_text()))
    with pytest.raises(ValueError, match="nonempty"):
        worker.run(CONFIG, reference, output)
    assert json.loads((output / "result.json").read_text()) == first
