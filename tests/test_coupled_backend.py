"""Transactional controller recovery; fake dynamics test mechanics, not TORAX."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfmarl.agents.beam_sac import SAC, SACConfig
from hfmarl.agents.thermal_td3 import TD3Config
from hfmarl.devices.registry import get
from hfmarl.envs.deposition_env import DepositionReward, PowerBudgetAllocator, sparc_deposition_task
from hfmarl.envs.limits import LimitSet
from hfmarl.experiments.coupled_control import (
    CoupledBackend, CoupledConfig, environment_contract, physical_snapshot,
)


class MechanicsEnv:
    def __init__(self):
        self.device = get("sparc_like")
        self.task = sparc_deposition_task(episode_length=2.)
        self.reward_spec = DepositionReward()
        self.target_Te_keV = np.ones(8)
        self.calibration_mode = False
        self.allocator = PowerBudgetAllocator(self.device)
        self.core = SimpleNamespace(config_dict={"transport": {"model_name": "constant"}})
        self.limits = LimitSet()
        self.limit_penalty, self.violation_penalty = 5., 50.
        self.strict_initial_state, self.action_cap = True, None

    def reset(self):
        self.step_index = 0
        return np.zeros(37, np.float32), {}

    def step(self, actions):
        self.step_index += 1
        obs = np.zeros(37, np.float32)
        obs[:4] = actions
        obs[8] = self.step_index / self.task.steps_per_shot
        return obs, -float(np.square(actions).sum()), False, self.step_index == self.task.steps_per_shot, {}


def backend(path, env=None, *, gate_reference="incumbent"):
    config = CoupledConfig(episodes_per_round=1, warmup_episodes=1,
                           gate_episodes=2, evaluation_episodes=1,
                           gate_reference=gate_reference)
    td3 = TD3Config(hidden=8, batch=2, head_steps=2, rep_steps=2,
                     repair_steps=2, adapt_steps=2)
    sac = SACConfig(hidden=8, batch=2, head_steps=2, rep_steps=2,
                     repair_steps=2, adapt_steps=2)
    return CoupledBackend("a", env or MechanicsEnv(), path, config, td3, sac, revision="fixed")


def frozen_state(value):
    """Compare full learning state without unstable pickle tensor-storage IDs."""
    if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
        return frozen_state(value.state_dict())
    if isinstance(value, torch.Generator):
        return frozen_state(value.get_state())
    if isinstance(value, np.random.Generator):
        return frozen_state(value.bit_generator.state)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, frozen_state(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(frozen_state(item) for item in value)
    return value


def pair_state(pair):
    return tuple(frozen_state(getattr(pair, role).__dict__) for role in ("heating", "beam"))


def assert_learner_steps(first, second):
    assert first.state["round"] == second.state["round"]
    for arm in first.state["learners"]:
        for name in ("heating", "beam"):
            a, b = getattr(first.state["learners"][arm], name), getattr(second.state["learners"][arm], name)
            assert (a.steps, a.env_steps, len(a.replay)) == (b.steps, b.env_steps, len(b.replay))
            for key, value in a.actor.state_dict().items():
                np.testing.assert_array_equal(value.detach().numpy(), b.actor.state_dict()[key].detach().numpy())


@pytest.mark.parametrize("field", ["profile", "transport", "radius", "reward", "task", "limits"])
def test_environment_fingerprint_binds_physics_even_with_unchanged_revision(tmp_path, field):
    original = backend(tmp_path)
    changed = MechanicsEnv()
    if field == "profile":
        changed.target_Te_keV *= 1.1
    elif field == "transport":
        changed.core.config_dict["transport"]["model_name"] = "changed"
    elif field == "radius":
        changed.allocator = PowerBudgetAllocator(changed.device, rho_bounds=(.05, .9))
    elif field == "reward":
        changed.reward_spec = DepositionReward(profile_weight=2.)
    elif field == "task":
        changed.task = replace(changed.task, disturbance_std=.1)
    else:
        changed.limits = LimitSet((replace(changed.limits.limits[0], hard=.9), *changed.limits.limits[1:]))
    assert environment_contract(changed)["sha256"] != original.contract["environment"]["sha256"]
    with pytest.raises(ValueError, match="changed client contract"):
        backend(tmp_path, changed)


@pytest.mark.parametrize("gate_reference", ["incumbent", "repaired_control"])
def test_deterministic_gate_fails_closed_and_extra_optimizer_work_is_counted(tmp_path, monkeypatch, gate_reference):
    model = backend(tmp_path, gate_reference=gate_reference)
    bases = model.state["learners"]["raw_federated"].representations()
    model.contribute(bases, 1)
    def forbidden_bootstrap(*args, **kwargs):
        raise AssertionError("Repeated deterministic trajectories are not a statistical gate")
    monkeypatch.setattr("hfmarl.experiments.coupled_control.acceptance", forbidden_bootstrap)
    report = model.accept(bases, 1)
    assert report["gate"] == {"accept": False, "reason": "no_independent_gate_cases",
                              "scope": "mechanics_only; deterministic repeated trajectories"}
    assert report["gate_episodes"] == 4
    assert report["gate_reference"] == gate_reference
    assert report["evaluation_episodes"] == report["training_episodes"] == 5
    for work in report["contribution_optimization"].values():
        assert work == {"head_sgd_updates": 2, "trunk_sgd_updates": 2, "replay_samples": 8}
    assert set(report["repair_optimization"]) == {"reset_only", "raw_federated", "gate_control", "gate_candidate"}
    for arm, work in report["repair_optimization"].items():
        if arm == "gate_control" and gate_reference == "incumbent":
            assert all(value == 0 for role in work.values() for value in role.values())
            continue
        assert work["heating_td3"]["actor_adaptation_updates"] == 1
        assert work["deposition_sac"]["actor_adaptation_updates"] == 2
        assert work["deposition_sac"]["entropy_temperature_updates"] == 2
        assert work["heating_td3"]["entropy_temperature_updates"] == 0
        assert all(role["replay_samples"] == 8 for role in work.values())


def test_rejected_candidate_preserves_every_incumbent_learning_state(tmp_path, monkeypatch):
    model = backend(tmp_path)
    bases = model.state["learners"]["raw_federated"].representations()
    model.contribute(bases, 1)
    incumbent = model.state["learners"]["deployed"]
    before = pair_state(incumbent)
    monkeypatch.setattr(model, "_gate_decision", lambda *args: {"accept": False, "reason": "test_rejection"})
    report = model.accept(bases, 1)
    assert model.state["learners"]["deployed"] is incumbent
    assert pair_state(model.state["learners"]["deployed"]) == before
    assert pair_state(backend(tmp_path).state["learners"]["deployed"]) == before
    assert all(value == 0 for role in report["repair_optimization"]["gate_control"].values()
               for value in role.values())
    assert report["repair_optimization"]["gate_candidate"]["deposition_sac"]["actor_adaptation_updates"] == 2


def test_accepted_candidate_deploys_the_evaluated_repaired_learner(tmp_path, monkeypatch):
    model = backend(tmp_path)
    bases = model.state["learners"]["raw_federated"].representations()
    model.contribute(bases, 1)
    incumbent = model.state["learners"]["deployed"]
    before = pair_state(incumbent)
    evaluated = []
    evaluate = model._evaluate
    def capture(learner, round_id, *, gate=False):
        if gate:
            evaluated.append(learner)
        return evaluate(learner, round_id, gate=gate)
    monkeypatch.setattr(model, "_evaluate", capture)
    monkeypatch.setattr(model, "_gate_decision", lambda *args: {"accept": True, "reason": "test_acceptance"})
    model.accept(bases, 1)
    assert evaluated[0] is incumbent
    assert model.state["learners"]["deployed"] is evaluated[1]
    assert pair_state(incumbent) == before
    assert pair_state(evaluated[1]) != before
    assert evaluated[1].heating.steps == incumbent.heating.steps + 2
    assert evaluated[1].beam.steps == incumbent.beam.steps + 2


def test_legacy_gate_rejection_still_deploys_repaired_control(tmp_path, monkeypatch):
    model = backend(tmp_path, gate_reference="repaired_control")
    bases = model.state["learners"]["raw_federated"].representations()
    model.contribute(bases, 1)
    incumbent = model.state["learners"]["deployed"]
    expected = pair_state(incumbent.repair())
    monkeypatch.setattr(model, "_gate_decision", lambda *args: {"accept": False, "reason": "test_rejection"})
    report = model.accept(bases, 1)
    deployed = model.state["learners"]["deployed"]
    assert deployed is not incumbent and pair_state(deployed) == expected
    assert report["gate_reference"] == "repaired_control"
    assert report["repair_optimization"]["gate_control"]["deposition_sac"]["actor_adaptation_updates"] == 2


def test_gate_reference_is_validated_and_bound_to_checkpoint_recovery(tmp_path):
    with pytest.raises(ValueError, match="gate reference"):
        CoupledConfig(gate_reference="unknown")
    model = backend(tmp_path, gate_reference="repaired_control")
    assert model.contract["config"]["gate_reference"] == "repaired_control"
    backend(tmp_path, gate_reference="repaired_control")
    with pytest.raises(ValueError, match="changed client contract"):
        backend(tmp_path, gate_reference="incumbent")


def test_export_failure_keeps_committed_learning_and_retry_regenerates_evidence(tmp_path, monkeypatch):
    model = backend(tmp_path)
    bases = model.state["learners"]["raw_federated"].representations()
    model.contribute(bases, 1)
    def failed_export(round_id):
        raise OSError("injected export failure")
    monkeypatch.setattr(model, "_export_record", failed_export)
    with pytest.raises(OSError, match="export failure"):
        model.accept(bases, 1)
    assert model.state["round"] == 1
    restored = backend(tmp_path)
    assert_learner_steps(model, restored)
    before = copy.deepcopy(restored)
    report = restored.accept(bases, 1)
    assert_learner_steps(before, restored)
    record = json.loads((tmp_path / "round-1.json").read_text(encoding="utf-8"))
    assert record["report"] == report
    assert restored.accept(bases, 1) == report
    assert_learner_steps(before, restored)


def test_failed_contribution_preserves_attempt_and_rolls_back_learning(tmp_path, monkeypatch):
    model = backend(tmp_path)
    bases = model.state["learners"]["raw_federated"].representations()
    untouched = copy.deepcopy(model)
    def failed_contribution(*args, **kwargs):
        raise ValueError("injected source failure")
    monkeypatch.setattr(SAC, "contribution", failed_contribution)
    with pytest.raises(ValueError, match="source failure"):
        model.contribute(bases, 1)
    assert_learner_steps(untouched, model)
    restored = backend(tmp_path)
    assert restored.state["fit"] is None
    assert restored.state["failures"] == [{
        "phase": "contribution", "round": 1, "source": "a",
        "error_type": "ValueError", "error": "injected source failure",
        "environment_sha256": restored.contract["environment"]["sha256"],
    }]


def test_physical_snapshots_keep_profiles_and_explicitly_flag_invalid_values():
    core = SimpleNamespace(t=np.float64(.5),
        read_scalars=lambda: {"beta_N": np.float64(.44)},
        read_profiles=lambda: {"rho_norm": np.array([.1, .9]),
                               "T_e": np.array([10., np.nan])})
    snapshot = physical_snapshot(SimpleNamespace(core=core))
    assert snapshot["t"] == .5 and snapshot["scalars"]["beta_N"] == .44
    assert snapshot["profiles"]["T_e"] == [10., None]
    assert snapshot["nonfinite_fields"] == ["profiles.T_e[1]"]
    json.dumps(snapshot, allow_nan=False)
    def failed_read():
        raise RuntimeError("failed solver state")
    core.read_profiles = failed_read
    snapshot = physical_snapshot(SimpleNamespace(core=core))
    assert "profiles" not in snapshot
    assert snapshot["read_errors"]["profiles"] == "RuntimeError: failed solver state"
    assert snapshot["scalars"] == {"beta_N": .44}
