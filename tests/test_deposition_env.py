"""Allocation, profile semantics, and opt-in live TORAX deposition checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from hfmarl.devices.registry import get
from hfmarl.envs.deposition_env import (
    PROFILE_RHO,
    CoupledDepositionEnv,
    PowerBudgetAllocator,
    _DepositionCore,
    sample_temperature_profiles,
    sparc_deposition_task,
)


def test_allocator_endpoints_and_partial_budget():
    allocator = PowerBudgetAllocator(get("sparc_like"))
    command = allocator.allocate([1, 0], [-1, 0])
    assert command.aux_power_W == 25e6
    assert command.ec_budget_W == 2.5e6
    assert command.ec_power_W == 1.25e6
    assert command.ec_rho == 0.05
    assert allocator.allocate([-1, -1], [1, 1]).ec_power_W == 0


def test_allocator_clips_finite_requests_but_preserves_raw_values():
    command = PowerBudgetAllocator(get("sparc_like")).allocate([3, -2], [-5, 6])
    assert command.requested_td3 == (3, -2)
    assert command.clipped_td3 == (1, -1)
    assert command.clipped_sac == (-1, 1)
    assert command.ec_power_W == 0


@pytest.mark.parametrize("bad", [[float("nan"), 0], [float("inf"), 0], [0], [[0, 0]]])
def test_nonfinite_or_malformed_actions_fail_closed(bad):
    allocator = PowerBudgetAllocator(get("sparc_like"))
    with pytest.raises(ValueError, match="finite vector"):
        allocator.allocate(bad, [0, 0])
    with pytest.raises(ValueError, match="finite vector"):
        allocator.allocate([0, 0], bad)


def test_delivery_never_exceeds_linearly_ramped_budget():
    allocator = PowerBudgetAllocator(get("sparc_like"))
    rng = np.random.default_rng(934)
    previous_power = previous_budget = 0.0
    for _ in range(200):
        command = allocator.allocate(rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2))
        for fraction in np.linspace(0, 1, 11):
            power = previous_power + fraction * (command.ec_power_W - previous_power)
            budget = previous_budget + fraction * (
                command.ec_budget_W - previous_budget
            )
            assert 0 <= power <= budget + 1e-8
        previous_power, previous_budget = command.ec_power_W, command.ec_budget_W


@pytest.mark.parametrize(
    "bounds", [(0.9, 0.1), (-1, 0.5), (0.1, 1.1), (0.1, float("nan"))]
)
def test_invalid_radius_envelopes_rejected(bounds):
    with pytest.raises(ValueError):
        PowerBudgetAllocator(get("sparc_like"), bounds)


def _profiles():
    rho = (np.arange(25) + 0.5) / 25
    return {"rho_norm": rho, "T_i": 8 - 5 * rho, "T_e": 10 - 6 * rho}


def test_profile_sampling_uses_physical_cell_centers_not_array_index():
    ti, te = sample_temperature_profiles(_profiles())
    np.testing.assert_allclose(ti, 8 - 5 * PROFILE_RHO)
    np.testing.assert_allclose(te, 10 - 6 * PROFILE_RHO)


@pytest.mark.parametrize(
    "change", ["nan", "negative", "missing", "wrong_grid", "short_grid"]
)
def test_invalid_profiles_are_not_silently_scored(change):
    profiles = _profiles()
    if change == "nan":
        profiles["T_e"][4] = np.nan
    elif change == "negative":
        profiles["T_i"][4] = -1
    elif change == "missing":
        del profiles["T_e"]
    elif change == "wrong_grid":
        profiles["rho_norm"] = profiles["rho_norm"][::-1]
    else:
        profiles["rho_norm"] = np.linspace(0.2, 0.8, 25)
    with pytest.raises(ValueError):
        sample_temperature_profiles(profiles)


class FakeCore:
    def __init__(self, *, fail=False):
        self.t = 0.0
        self.t_initial = 0.0
        self.t_final = 1.0
        self.delta_t_a = 0.5
        self.fail = fail
        self.updates = []

    def reset(self):
        self.t = 0.0

    def read_profiles(self):
        return _profiles()

    def read_scalars(self):
        return {"beta_N": 0.44, "q95": 4.0, "fgw_n_e_line_avg": 0.5}

    def limit_inputs(self):
        return {"beta_N": 0.44, "q95": 4.0, "greenwald_fraction": 0.5}

    def nonfinite_scalars(self):
        return ()

    def apply_updates(self, updates):
        self.updates.append(updates)

    def advance(self):
        if self.fail:
            return False, True, "fixture solver failure"
        self.t += self.delta_t_a
        return True, self.t >= self.t_final, ""


def _env(*, fail=False):
    env = CoupledDepositionEnv(
        get("sparc_like"),
        target_Te_keV=np.ones(8) * 5,
        task=sparc_deposition_task(episode_length=1),
    )
    env.core = FakeCore(fail=fail)
    return env


def test_initial_and_role_observations_contain_the_declared_state():
    env = _env()
    obs, info = env.reset()
    assert obs.shape == (37,)
    assert info["calibration_mode"] is False
    np.testing.assert_allclose(obs[9:17], (8 - 5 * PROFILE_RHO) / 10)
    np.testing.assert_allclose(obs[17:25], (10 - 6 * PROFILE_RHO) / 10)
    np.testing.assert_allclose(obs[25:33], 0.5)
    np.testing.assert_allclose(obs[33:], [0, 0, 0.1, 0])
    sac = env.sac_observation([0.2, 0.8])
    assert sac.shape == (39,)
    np.testing.assert_allclose(sac[-2:], [0.2, 0.8])


def test_step_writes_actual_ec_location_and_one_owner_per_source(fake_torax):
    env = _env()
    env.reset()
    _, reward, terminated, truncated, info = env.step([1, 0, 1, 0])
    assert not terminated and not truncated
    update = env.core.updates[0]
    assert set(update) == {
        "sources.generic_heat.P_total",
        "sources.ecrh.P_total",
        "sources.ecrh.gaussian_location",
    }
    np.testing.assert_allclose(
        update["sources.ecrh.gaussian_location"].value, [0.1, 0.85]
    )
    np.testing.assert_allclose(update["sources.ecrh.P_total"].value, [0, 1.25e6])
    np.testing.assert_allclose(update["sources.ecrh.P_total"].time, [0, 0.5])
    assert info["allocator"]["ec_budget_W"] == 2.5e6
    assert info["allocator"]["ec_power_W"] == 1.25e6
    assert info["shared_reward"] == reward < 0
    assert env.trajectory[0].actuators["ec_rho"] == pytest.approx(0.85)
    obs, _, _, truncated, info = env.step([-1, -1, -1, 1])
    assert truncated
    np.testing.assert_allclose(obs[33:], [0, 0, 0.05, 0])
    assert info["allocator"]["ec_budget_start_W"] == 2.5e6
    np.testing.assert_allclose(
        env.core.updates[1]["sources.ecrh.P_total"].value, [1.25e6, 0]
    )
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))


def test_invalid_action_does_not_mutate_allocator_or_advance():
    env = _env()
    env.reset()
    with pytest.raises(ValueError):
        env.step([0, 0, np.nan, 0])
    assert env.command_history == []
    assert env.core.updates == []
    assert env.last_ec_budget_W == 0


def test_solver_failure_charges_whole_remaining_horizon_and_preserves_attempt(
    fake_torax,
):
    env = _env(fail=True)
    initial, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated
    assert reward <= -env.violation_penalty - 0.5 * env.reward_spec.maximum_cost_rate
    assert info["allocator"]["completion"] == "failed"
    assert info["violations"] == ("solver_failure",)
    assert np.isfinite(obs).all()
    np.testing.assert_array_equal(obs, initial)
    np.testing.assert_allclose(env.bank.prev_values, [0, 0, 0.1])
    assert env.core.t == 0


def test_profile_failure_is_a_terminal_core_error():
    core = _DepositionCore({"numerics": {"fixed_dt": 0.1}}, 0.5)
    core.post = SimpleNamespace(beta_N=0.44)
    profiles = _profiles()
    profiles["T_e"][0] = np.nan
    core.read_profiles = lambda: profiles
    assert "temperature_profile_or_grid" in core.nonfinite_scalars()


def test_requires_frozen_profile_or_explicit_calibration_mode():
    with pytest.raises(ValueError, match="frozen calibrated"):
        CoupledDepositionEnv(get("sparc_like"))
    env = CoupledDepositionEnv(get("sparc_like"), calibration_mode=True)
    assert env.task.setpoint_mode == "absolute"
    assert env.task.setpoint.base == pytest.approx(0.4408455389299299)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_beta": 0.9},
        {"delta_t_a": 0.35},
        {"episode_length": np.nan},
        {"delta_t_a": 0},
    ],
)
def test_invalid_or_uncalibrated_default_tasks_rejected(kwargs):
    with pytest.raises(ValueError):
        sparc_deposition_task(**kwargs)


def test_reward_separates_pressure_profile_and_applied_power(fake_torax):
    env = _env()
    env.reset()
    _, _, _, _, info = env.step([-1, 1, 0, -1])
    cost = info["reward_components"]
    assert cost["beta_cost"] > 0
    assert cost["profile_cost"] > 0
    assert (
        cost["normalized_power"] == 0
    )  # reserved but unused EC costs no delivered power


def test_existing_thermal_environment_unchanged():
    from hfmarl.envs.torax_env import ToraxDeviceEnv

    env = ToraxDeviceEnv(get("sparc_like"))
    assert env.n_actions == 2
    assert env.bank.names == ("aux_heat", "ecrh")
    assert env.core.config_dict["sources"]["ecrh"]["gaussian_location"] == 0.1


def test_calibration_cli_preserves_incomplete_cases_and_source_archive(
    tmp_path, monkeypatch, fake_torax
):
    from scripts import calibrate_deposition

    def env_factory(device, **kwargs):
        # A fixture that truncates after 1 second when the declared calibration
        # horizon is 2 seconds must never produce a usable reference.
        return _env()

    monkeypatch.setattr(calibrate_deposition, "CoupledDepositionEnv", env_factory)
    # This fixture has no real config; the hash still needs a recorded input.
    original = env_factory

    def with_config(*args, **kwargs):
        env = original(*args, **kwargs)
        env.core.config_dict = {"fixture": True}
        return env

    monkeypatch.setattr(calibrate_deposition, "CoupledDepositionEnv", with_config)
    output = tmp_path / "calibration"
    summary = calibrate_deposition.run(output, episode_length=2.0)
    assert summary["completed_cases"] == 0
    assert summary["attempted_windows"] == 12
    assert not (output / "reference.json").exists()
    assert (output / "source.zip").is_file()
    assert len(summary["timestep_pairs"]) == 3
    assert all(
        "incomplete_case" in pair["flagged_samples"]
        for pair in summary["timestep_pairs"]
    )


@pytest.mark.torax
def test_live_ec_deposition_changes_electron_profile_and_runtime_location():
    pytest.importorskip("torax")
    task = sparc_deposition_task(episode_length=1.0)
    env = CoupledDepositionEnv(get("sparc_like"), task=task, calibration_mode=True)
    profiles = []
    for radius_action, expected_rho in [(-0.625, 0.2), (0.625, 0.7)]:
        env.reset()
        for _ in range(task.steps_per_shot):
            _, _, terminated, _, info = env.step([0, 1, radius_action, 1])
            assert not terminated, info
        runtime = env.core.step_fn.runtime_params_provider(env.core.t)
        assert float(runtime.sources["ecrh"].gaussian_location) == pytest.approx(
            expected_rho
        )
        profiles.append(env.core.read_profiles()["T_e"].copy())
    assert np.max(np.abs(profiles[0] - profiles[1])) > 1e-4
