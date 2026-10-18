"""Opt-in coupled heating and EC deposition control in live TORAX.

TD3 requests auxiliary power and an EC power budget. SAC chooses the EC
deposition radius and the fraction of that budget to use. There is exactly one
writer for each TORAX source parameter. This is prescribed Gaussian deposition,
not ray tracing, mirror steering or magnetic-island suppression.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from hfmarl.devices.registry import Device
from hfmarl.envs.actuators import ActuatorBank, ActuatorSpec
from hfmarl.envs.task import SetpointSchedule, TaskSpec
from hfmarl.envs.torax_env import ToraxCore, ToraxDeviceEnv

PROFILE_RHO = np.linspace(0.08, 0.92, 8)
PROFILE_RHO.setflags(write=False)
OBSERVATION_SCHEMA = (
    "coupled-deposition-v1:base9:Ti8-Te8-targetTe8-keV10:applied3:budget1"
)
SPARC_BETA_SCALE = 0.4789417901771832 - 0.40274928768267665


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result


@dataclass(frozen=True)
class AllocatedCommand:
    requested_td3: tuple[float, float]
    requested_sac: tuple[float, float]
    clipped_td3: tuple[float, float]
    clipped_sac: tuple[float, float]
    aux_power_W: float
    ec_budget_W: float
    ec_fraction: float
    ec_power_W: float
    ec_rho: float

    @property
    def applied_values(self) -> np.ndarray:
        return np.array([self.aux_power_W, self.ec_power_W, self.ec_rho])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PowerBudgetAllocator:
    """Pure deterministic arbitration; rejects NaN/inf before changing state.

    Budget and delivered EC power use the same linear ramp between windows.
    Endpoint inequalities therefore imply delivered power <= budget throughout
    the window. A reduction in budget is a ramp, not an instantaneous ceiling.
    """

    def __init__(self, device: Device, rho_bounds=(0.05, 0.85)):
        self.aux = device.actuator("aux_heat")
        self.ec = device.actuator("ecrh")
        self.rho_bounds = tuple(float(x) for x in rho_bounds)
        if (
            len(self.rho_bounds) != 2
            or not np.isfinite(self.rho_bounds).all()
            or not 0 <= self.rho_bounds[0] < self.rho_bounds[1] <= 1
        ):
            raise ValueError("rho_bounds must be finite and satisfy 0 <= lo < hi <= 1")
        for actuator in (self.aux, self.ec):
            if (
                not actuator.available
                or actuator.lo != 0
                or not np.isfinite(actuator.hi)
                or actuator.hi <= 0
            ):
                raise ValueError(
                    "coupled heating requires available zero-to-positive power actuators"
                )
        if self.aux.torax_path == self.ec.torax_path:
            raise ValueError(
                "auxiliary and EC power must have distinct TORAX source paths"
            )

    def allocate(self, td3_action, sac_action) -> AllocatedCommand:
        raw_td3 = _finite_vector(td3_action, 2, "TD3 action")
        raw_sac = _finite_vector(sac_action, 2, "SAC action")
        td3 = np.clip(raw_td3, -1, 1)
        sac = np.clip(raw_sac, -1, 1)
        aux = (td3[0] + 1) * 0.5 * self.aux.hi
        budget = (td3[1] + 1) * 0.5 * self.ec.hi
        fraction = (sac[1] + 1) * 0.5
        rho = self.rho_bounds[0] + (sac[0] + 1) * 0.5 * np.ptp(self.rho_bounds)
        return AllocatedCommand(
            tuple(raw_td3),
            tuple(raw_sac),
            tuple(td3),
            tuple(sac),
            float(aux),
            float(budget),
            float(fraction),
            float(budget * fraction),
            float(rho),
        )


@dataclass(frozen=True)
class DepositionReward:
    beta_scale: float = SPARC_BETA_SCALE
    profile_scale_keV: float = 1.0
    profile_weight: float = 1.0
    power_weight: float = 0.01
    slew_weight: float = 0.01
    maximum_cost_rate: float = 25.0

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("reward parameters must be finite")
        if min(self.beta_scale, self.profile_scale_keV, self.maximum_cost_rate) <= 0:
            raise ValueError("reward scales and maximum cost rate must be positive")
        if min(self.profile_weight, self.power_weight, self.slew_weight) < 0:
            raise ValueError("reward weights must be nonnegative")


def sparc_deposition_task(
    *, target_beta=0.4408455389299299, episode_length=15.0, delta_t_a=0.5
) -> TaskSpec:
    """Absolute corrected-density calibration target; no legacy band lookup.

    This pressure target came from the fixed-deposition calibration. A coupled
    reference run must still establish its joint feasibility with a Te profile.
    """
    if not 0.42179741330630327 <= target_beta <= 0.45989366455355657:
        raise ValueError(
            "SPARC target must be inside the corrected calibration's interior band"
        )
    if (
        not np.isfinite([episode_length, delta_t_a]).all()
        or min(episode_length, delta_t_a) <= 0
        or not np.isclose(
            episode_length / delta_t_a,
            round(episode_length / delta_t_a),
            rtol=0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "episode length must be a positive integer number of action windows"
        )
    return TaskSpec(
        name="sparc_coupled_deposition",
        setpoint=SetpointSchedule(base=target_beta),
        tolerance=0.005,
        transport_model="constant",
        enable_fusion=True,
        episode_length=episode_length,
        delta_t_a=delta_t_a,
        setpoint_mode="absolute",
        tolerance_mode="absolute",
        disturbance_std=0,
    )


def sample_temperature_profiles(
    profiles: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample real cell profiles on a fixed physical normalized-radius grid."""
    rho = np.asarray(profiles.get("rho_norm", []), dtype=float)
    if (
        rho.ndim != 1
        or rho.size < 2
        or not np.isfinite(rho).all()
        or not np.all(np.diff(rho) > 0)
        or rho[0] > PROFILE_RHO[0]
        or rho[-1] < PROFILE_RHO[-1]
    ):
        raise ValueError(
            "valid TORAX rho_norm cell grid covering the observation grid is required"
        )
    result = []
    for key in ("T_i", "T_e"):
        value = np.asarray(profiles.get(key, []), dtype=float)
        if (
            value.shape != rho.shape
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise ValueError(
                f"{key} must be a finite nonnegative profile on the TORAX cell grid"
            )
        result.append(np.interp(PROFILE_RHO, rho, value))
    return tuple(result)


class _DepositionCore(ToraxCore):
    def read_profiles(self):
        out = super().read_profiles()
        geo = self.step_fn.geometry_provider(self.t)
        out["rho_norm"] = np.asarray(geo.rho_norm, dtype=float)
        return out

    def nonfinite_scalars(self):
        bad = list(super().nonfinite_scalars())
        scalars = self.read_scalars()
        for name in ("li3", "H98", "f_bootstrap"):
            if name in scalars and not np.isfinite(scalars[name]):
                bad.append(name)
        try:
            sample_temperature_profiles(self.read_profiles())
        except (ValueError, AttributeError):
            bad.append("temperature_profile_or_grid")
        return tuple(bad)


class CoupledDepositionEnv(ToraxDeviceEnv):
    """Shared 37-value observation and four normalized role actions.

    ``step([aux, budget, rho, fraction])`` and ``step_roles(td3, sac)`` are
    equivalent. ``sac_observation(td3)`` appends the current allocation proposal
    (39 values), so SAC can condition its decision on this window's budget.
    Replay must retain that proposal, including the *next* proposal used by the
    behavior policy. An old thermal TD3 adapter cannot interpret this schema.
    """

    def __init__(
        self,
        device: Device,
        *,
        target_Te_keV=None,
        calibration_mode: bool = False,
        reward: DepositionReward | None = None,
        rho_bounds=(0.05, 0.85),
        task: TaskSpec | None = None,
        **kwargs,
    ):
        if type(calibration_mode) is not bool:
            raise TypeError("calibration_mode must be an explicit bool")
        if target_Te_keV is None and not calibration_mode:
            raise ValueError(
                "a frozen calibrated target_Te_keV is required for training"
            )
        self.target_Te_keV = (
            None
            if target_Te_keV is None
            else _finite_vector(target_Te_keV, len(PROFILE_RHO), "target_Te_keV").copy()
        )
        if self.target_Te_keV is not None and np.any(self.target_Te_keV <= 0):
            raise ValueError("target_Te_keV must be positive")
        if self.target_Te_keV is not None:
            self.target_Te_keV.setflags(write=False)
        self.calibration_mode = calibration_mode
        self.reward_spec = reward or DepositionReward()
        self.allocator = PowerBudgetAllocator(device, rho_bounds)
        if task is None:
            if device.name != "sparc_like":
                raise ValueError(
                    "non-SPARC devices require an explicitly calibrated TaskSpec"
                )
            task = sparc_deposition_task()
        if (
            not np.isfinite([task.delta_t_a, task.episode_length]).all()
            or min(task.delta_t_a, task.episode_length) <= 0
            or not np.isclose(
                task.episode_length / task.delta_t_a,
                round(task.episode_length / task.delta_t_a),
                rtol=0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "episode length must be a positive integer number of action windows"
            )
        if kwargs.get("action_cap") is not None:
            raise ValueError(
                "action_cap is ambiguous for role actions; use explicit allocator envelopes"
            )
        super().__init__(device, task=task, clusters=("thermal",), **kwargs)
        if (
            not np.isfinite([self.violation_penalty, self.limit_penalty]).all()
            or self.limit_penalty < 0
            or self.violation_penalty
            <= self.reward_spec.maximum_cost_rate * task.delta_t_a
        ):
            raise ValueError(
                "violation penalty must exceed one window's maximum running cost"
            )
        config = self.core.config_dict
        initial_rho = float(config["sources"]["ecrh"]["gaussian_location"])
        if not rho_bounds[0] <= initial_rho <= rho_bounds[1]:
            raise ValueError("rho bounds must include the TORAX initial EC location")
        self.bank = ActuatorBank(
            [
                ActuatorSpec(
                    "aux_heat",
                    self.allocator.aux.torax_path,
                    0,
                    self.allocator.aux.hi,
                    "thermal",
                    "W",
                ),
                ActuatorSpec(
                    "ecrh",
                    self.allocator.ec.torax_path,
                    0,
                    self.allocator.ec.hi,
                    "deposition",
                    "W",
                ),
                ActuatorSpec(
                    "ec_rho",
                    "sources.ecrh.gaussian_location",
                    *rho_bounds,
                    "deposition",
                    "normalized radius",
                    initial=initial_rho,
                ),
            ]
        )
        self.core = _DepositionCore(config, task.delta_t_a)
        self.last_ec_budget_W = 0.0
        self._last_applied = self.bank.prev_values.copy()
        self._last_slew = 0.0
        self._last_observation = np.zeros(self.observation_size, dtype=np.float32)
        self.command_history: list[dict] = []
        self.reward_components: dict[str, float] = {}

    @property
    def n_actions(self):
        return 4

    @property
    def observation_size(self):
        return 37

    def reset(self):
        self.last_ec_budget_W = 0.0
        self._last_slew = 0.0
        self._last_applied = np.array([s.start_value for s in self.bank.specs])
        self.command_history = []
        self.reward_components = {}
        self._last_observation = np.zeros(self.observation_size, dtype=np.float32)
        obs, info = super().reset()
        info.update(
            observation_schema=OBSERVATION_SCHEMA,
            calibration_mode=self.calibration_mode,
            profile_rho=PROFILE_RHO.tolist(),
        )
        return obs, info

    def sac_observation(self, td3_action):
        td3 = np.clip(_finite_vector(td3_action, 2, "TD3 action"), -1, 1)
        return np.concatenate((self._observe(), td3)).astype(np.float32)

    def step(self, action):
        action = _finite_vector(action, 4, "joint action")
        return self.step_roles(action[:2], action[2:])

    def step_roles(self, td3_action, sac_action):
        if not self._episode_ready:
            raise RuntimeError(
                "A successful reset is required before stepping an episode"
            )
        command = self.allocator.allocate(td3_action, sac_action)
        previous = self.bank.prev_values.copy()
        previous_observation = self._last_observation.copy()
        budget_start = self.last_ec_budget_W
        scales = np.array([self.allocator.aux.hi, self.allocator.ec.hi, 1.0])
        self._last_slew = float(
            np.mean(((command.applied_values - previous) / scales) ** 2)
        )
        self.last_ec_budget_W = command.ec_budget_W
        self._last_applied = command.applied_values
        record = command.to_dict()
        record.update(
            t_start=float(self.core.t),
            t_end=float(self.core.t + self.core.delta_t_a),
            ec_budget_start_W=budget_start,
            applied_start=previous.tolist(),
            applied_end=command.applied_values.tolist(),
            interpolation="linear",
            completion="attempted",
        )
        self.command_history.append(record)
        # Convert the already allocated physical commands into the three bank
        # coordinates. The base env writes the ramp and advances TORAX once.
        bank_action = np.array(
            [s.to_unit(v) for s, v in zip(self.bank.specs, command.applied_values)]
        )
        obs, shared, terminated, truncated, info = super().step(bank_action)
        record["completion"] = "failed" if info.get("error") else "advanced"
        if info.get("error"):
            # No valid endpoint was observed. Preserve the attempted command
            # above, but do not call its unobserved endpoint a delivered action.
            self._last_applied = previous
            self.last_ec_budget_W = budget_start
            self.bank.prev_values = previous.copy()
            self._last_observation = previous_observation
            obs = previous_observation.copy()
        info.update(
            allocator=record,
            shared_reward=shared,
            reward_components=dict(self.reward_components),
            calibration_mode=self.calibration_mode,
        )
        return obs, shared, terminated, truncated, info

    def _observe(self):
        # Invalid states terminate via the core check; replay receives the last
        # finite observation and a terminal flag, never fabricated valid data.
        try:
            ti, te = sample_temperature_profiles(self.core.read_profiles())
            base = super()._observe()
            target = np.zeros(8) if self.target_Te_keV is None else self.target_Te_keV
            applied = self._last_applied / np.array(
                [self.allocator.aux.hi, self.allocator.ec.hi, 1.0]
            )
            obs = np.concatenate(
                (
                    base,
                    ti / 10,
                    te / 10,
                    target / 10,
                    applied,
                    [self.last_ec_budget_W / self.allocator.ec.hi],
                )
            ).astype(np.float32)
            if not np.isfinite(obs).all():
                raise ValueError("nonfinite observation")
            self._last_observation = obs
        except (ValueError, AttributeError):
            if self._episode_ready:
                raise
        return self._last_observation.copy()

    def _control_cost(self, report):
        spec = self.reward_spec
        beta = self.core.read_scalars()["beta_N"]
        _, te = sample_temperature_profiles(self.core.read_profiles())
        tracking = ((beta - self.target_beta_N) / spec.beta_scale) ** 2
        profile = (
            0.0
            if self.target_Te_keV is None
            else float(
                np.mean(((te - self.target_Te_keV) / spec.profile_scale_keV) ** 2)
            )
        )
        power = (self._last_applied[0] + self._last_applied[1]) / (
            self.allocator.aux.hi + self.allocator.ec.hi
        )
        soft = self.limit_penalty * max(0.0, 1.0 - report.worst_margin) ** 2
        self.reward_components = {
            "beta_cost": tracking,
            "profile_cost": profile,
            "normalized_power": power,
            "normalized_slew": self._last_slew,
            "soft_limit_cost": soft,
        }
        rate = (
            tracking
            + spec.profile_weight * profile
            + spec.power_weight * power
            + spec.slew_weight * self._last_slew
            + soft
        )
        return -float(min(rate, spec.maximum_cost_rate) * self.core.delta_t_a)

    def _early_exit_charge(self):
        # Charge the maximum possible remaining running cost. Unlike billing
        # the last low-error state, a crash cannot escape a harder future target.
        remaining = max(0, self.task.steps_per_shot - len(self.trajectory) - 1)
        return -remaining * self.reward_spec.maximum_cost_rate * self.core.delta_t_a


@dataclass(frozen=True)
class DepositionReference:
    """Measured endpoint target and its explicit calibration cost/provenance."""

    target_Te_keV: tuple[float, ...]
    beta_N: float
    profile_rho: tuple[float, ...]
    joint_action: tuple[float, ...]
    config_sha256: str
    calibration_windows: int
    calibration_simulated_seconds: float
    device: str
    sampled_violations: tuple[str, ...]
    scope: str = "observed fixed-command endpoint; not whole-trajectory feasibility"

    def to_dict(self):
        return asdict(self)


def calibrate_deposition_reference(
    device: Device,
    *,
    joint_action=(0.0, 0.0, 0.0, 1.0),
    task: TaskSpec | None = None,
    config_overrides: dict | None = None,
) -> DepositionReference:
    """Run one bounded fixed-command shot and freeze its observed Te endpoint.

    The caller must preserve this artifact *before* training and separately
    count this run. Failures raise; no target is fabricated from an invalid shot.
    A paired smaller solver timestep and actuator sweep remain necessary for
    numerical validation and evidence that SAC has useful control authority.
    """
    action = _finite_vector(joint_action, 4, "joint action")
    env = CoupledDepositionEnv(
        device, calibration_mode=True, task=task, config_overrides=config_overrides
    )
    config_hash = hashlib.sha256(
        json.dumps(env.core.config_dict, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    env.reset()
    for _ in range(env.task.steps_per_shot):
        _, _, terminated, truncated, info = env.step(action)
        if terminated:
            raise RuntimeError(f"deposition calibration failed: {info}")
        if truncated:
            break
    if len(env.trajectory) != env.task.steps_per_shot or not truncated:
        raise RuntimeError(
            "deposition calibration did not complete the declared horizon"
        )
    _, te = sample_temperature_profiles(env.core.read_profiles())
    return DepositionReference(
        tuple(te),
        float(env.core.read_scalars()["beta_N"]),
        tuple(PROFILE_RHO),
        tuple(action),
        config_hash,
        len(env.trajectory),
        float(env.core.t - env.core.t_initial),
        device.name,
        (),
    )
