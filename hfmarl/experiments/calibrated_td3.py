"""Corrected-density, causal SPARC TD3 versus calibrated conventional control.

This is a development experiment, not an efficacy result. It intentionally
reuses the frozen calibration and tracking protocol instead of registry targets.
Every action uses only the target at the current observation time.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import tarfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from hfmarl.agents.thermal_td3 import TD3, TD3Config
from scripts import sparc_tracking_test as tracking

calibration = tracking.calibration
ROOT = Path(__file__).resolve().parents[2]
WINDOW, HORIZON, WINDOWS = tracking.WINDOW, tracking.HORIZON, 30
CONVENTIONAL = ("feedforward", "pi_feedback")
OBS_SCHEMA = "sparc-calibrated-td3-v1:beta,error,current_target,q95,fgw,previous2,time,energy"


@dataclass(frozen=True)
class BaselineConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    training_episodes: int = 120
    warmup_steps: int = 300
    training_dt: float = .05
    evaluation_dts: tuple[float, ...] = (.1, .05)
    evaluation_factors: tuple[float, ...] = tracking.TEMPERATURE_FACTORS
    schedules: tuple[str, ...] = tracking.TARGET_SCHEDULES
    td3: TD3Config = field(default_factory=lambda: TD3Config(
        batch=64, hidden=64, gamma=.95, reward_scale=1., actor_lr=1e-4,
        critic_head_init="zero", actor_init_fraction=(.5, .5)))

    def __post_init__(self):
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)):
            raise ValueError("Seeds must be unique nonnegative integers")
        if type(self.training_episodes) is not int or self.training_episodes < 1:
            raise ValueError("At least one training episode is required")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("Warmup steps must be a nonnegative integer")
        if self.training_dt not in (.1, .05) or not self.evaluation_dts:
            raise ValueError("Use the calibrated internal timesteps .1/.05")
        if (len(set(self.evaluation_dts)) != len(self.evaluation_dts)
                or any(dt not in (.1, .05) for dt in self.evaluation_dts)):
            raise ValueError("Evaluation timesteps must be unique .1/.05 values")
        if (not self.evaluation_factors or len(set(self.evaluation_factors)) != len(self.evaluation_factors)
                or any(not np.isfinite(x) or not .5 <= x <= 1.5 for x in self.evaluation_factors)):
            raise ValueError("Evaluation factors must be unique and within [.5, 1.5]")
        if (not self.schedules or len(set(self.schedules)) != len(self.schedules)
                or not set(self.schedules) <= set(tracking.TARGET_SCHEDULES)):
            raise ValueError("Use unique calibrated target schedules")
        if self.td3.batch < 1 or self.td3.reward_scale != 1.:
            raise ValueError("A positive batch and reward_scale=1 are required")


def case_definition(factor, schedule, dt):
    return {"device": "sparc_like", "normalize_n_e_to_nbar": True,
            "initial_temperature_factor": float(factor),
            "target_schedule": schedule, "fixed_dt": float(dt)}


def evaluation_cases(config):
    return [case_definition(factor, schedule, dt) for dt in config.evaluation_dts
            for factor in config.evaluation_factors for schedule in config.schedules]


def training_cases(config, seed):
    # This stream is independent of actor exploration and replay draws.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x43415345]))
    return [case_definition(rng.uniform(.9, 1.1), str(rng.choice(config.schedules)), config.training_dt)
            for _ in range(config.training_episodes)]


def jobs(config):
    cases = evaluation_cases(config)
    result = [{"phase": "evaluation", "controller": arm, "seed": None,
               "episode": i, "case": case} for arm in CONVENTIONAL for i, case in enumerate(cases)]
    for seed in config.seeds:
        result.extend({"phase": "initial_evaluation", "controller": "td3", "seed": seed,
                       "episode": i, "case": case} for i, case in enumerate(cases))
        result.extend({"phase": "training", "controller": "td3", "seed": seed,
                       "episode": i, "case": case} for i, case in enumerate(training_cases(config, seed)))
        result.extend({"phase": "evaluation", "controller": "td3", "seed": seed,
                       "episode": i, "case": case} for i, case in enumerate(cases))
    return [{"index": i, **job} for i, job in enumerate(result)]


def expected_budget(config):
    evaluation = len(evaluation_cases(config)) * (len(CONVENTIONAL) + len(config.seeds))
    initial_evaluation = len(evaluation_cases(config)) * len(config.seeds)
    training = len(config.seeds) * config.training_episodes
    all_jobs = jobs(config)
    return {"training_episodes": training, "evaluation_episodes": evaluation,
            "initial_evaluation_episodes": initial_evaluation,
            "total_episodes": training + evaluation + initial_evaluation,
            "maximum_action_windows": WINDOWS * (training + evaluation + initial_evaluation),
            "maximum_nominal_internal_steps": sum(round(HORIZON / job["case"]["fixed_dt"]) for job in all_jobs),
            "maximum_critic_updates": training * WINDOWS,
            "prior_calibration_episodes": 120,
            "prior_tracking_development_episodes": 180}


def observation(point, bank, target):
    """Current target only; no schedule name, next target, or future trajectory."""
    s = point["scalars"]
    previous = [(value - spec.lo) / (spec.hi - spec.lo)
                for value, spec in zip(bank.prev_values, bank.specs)]
    result = np.asarray([s["beta_N"] / tracking.MID,
                         (s["beta_N"] - target) / tracking.ERROR_SCALE,
                         (target - tracking.MID) / tracking.ERROR_SCALE,
                         s["q95"] / 10, s["fgw_n_e_line_avg"], *previous,
                         point["t"] / HORIZON, s["W_thermal_total"] / 1e8], np.float32)
    if result.shape != (9,) or not np.isfinite(result).all():
        raise ValueError("Invalid current observation")
    return result


def tracking_reward(error, failed, completed_windows):
    """Bounded normalized tracking; a failed remainder cannot improve return."""
    if not 0 <= completed_windows <= WINDOWS:
        raise ValueError("Invalid completed-window count")
    if failed:
        return float(-100 * (WINDOWS - completed_windows + 1) - 1)
    if error is None or not np.isfinite(error):
        raise ValueError("Finite tracking error required")
    return -min((float(error) / tracking.ERROR_SCALE) ** 2, 100.)


def new_agent(config, seed):
    bank = calibration.bank_from_device(calibration.get("sparc_like"), clusters=("thermal",))
    lo = np.array([s.lo * s.scale / 1e8 for s in bank.specs], np.float32)
    hi = np.array([s.hi * s.scale / 1e8 for s in bank.specs], np.float32)
    return TD3(9, lo, hi, seed, config.td3)


def policy_hash(agent):
    digest = hashlib.sha256()
    for name, tensor in agent.actor.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, agent, identity, seed, completed_training_episodes):
    payload = {"identity": identity, "config": asdict(agent.cfg),
               "seed": seed, "completed_training_episodes": completed_training_episodes,
               "actor": agent.actor.state_dict(), "critics": agent.critics.state_dict(),
               "target_actor": agent.target_actor.state_dict(),
               "target_critics": agent.target_critics.state_dict(),
               "actor_opt": agent.actor_opt.state_dict(), "critic_opt": agent.critic_opt.state_dict(),
               "replay": agent.replay, "cursor": agent.cursor, "steps": agent.steps,
               "env_steps": agent.env_steps, "rng": agent.rng.bit_generator.state,
               "replay_rng": agent.replay_rng.bit_generator.state,
               "torch_rng": agent.torch_rng.get_state()}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"seed": seed, "completed_training_episodes": completed_training_episodes,
            "sha256": calibration.sha(path.read_bytes()), "policy_sha256": policy_hash(agent)}


def load_checkpoint(path, agent, identity, seed, completed_training_episodes, expected_sha256=None):
    # Only load locally generated checkpoints from this study; pickle is not a
    # safe format for untrusted downloaded model files.
    if expected_sha256 is not None and calibration.sha(path.read_bytes()) != expected_sha256:
        raise ValueError("Checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["identity"] != identity or payload["config"] != asdict(agent.cfg):
        raise ValueError("Checkpoint identity/config mismatch")
    if payload.get("seed") != seed or payload.get("completed_training_episodes") != completed_training_episodes:
        raise ValueError("Checkpoint seed/training boundary mismatch")
    for name in ("actor", "critics", "target_actor", "target_critics", "actor_opt", "critic_opt"):
        getattr(agent, name).load_state_dict(payload[name])
    for name in ("replay", "cursor", "steps", "env_steps"):
        setattr(agent, name, payload[name])
    agent.rng.bit_generator.state = payload["rng"]
    agent.replay_rng.bit_generator.state = payload["replay_rng"]
    agent.torch_rng.set_state(payload["torch_rng"])
    return agent


def run_episode(job, config, save, agent=None):
    """Preserve every attempted window, failure and update; never retry a shot."""
    case = {**job["case"], "index": job["index"], "controller": job["controller"]}
    training = job["phase"] == "training"
    if training and (agent is None or job["controller"] != "td3"):
        raise ValueError("Training requires a TD3 learner")
    row = {"job": job, "case": case, "outcome": "running", "trace": [], "commands": [],
           "attempted_action_windows": 0, "completed_action_windows": 0,
           "initialization_completed": False, "pending_operation": "initialize",
           "failure": None, "learner_updates": [], "rewards": [],
           "learner_state_before": None if agent is None else {"updates": agent.steps,
                "environment_steps": agent.env_steps, "replay_size": len(agent.replay)},
           "policy_hash_before": None if agent is None else policy_hash(agent)}
    save(row)
    started = time.perf_counter()
    try:
        raw_config = tracking.case_config(case)
        bank = calibration.bank_from_device(calibration.get(case["device"]), clusters=("thermal",))
        core = calibration.ToraxCore(raw_config, delta_t_a=WINDOW)
        core.reset()
        row["initialization_completed"] = True
        row["resolved_torax_config"] = calibration.json_value(core.torax_config)
        row["resolved_torax_config_sha256"] = calibration.sha(calibration.canonical(row["resolved_torax_config"]))
        row["trace"].append(tracking.annotated_sample(core, bank, case, 0.))
        row["outcome"] = calibration.sampled_outcome(row["trace"][-1])
        row["pending_operation"] = None
        save(row)
        integral = 0.
        if row["outcome"] == "running":
            for index in range(WINDOWS):
                start, end = index * WINDOW, (index + 1) * WINDOW
                if abs(core.t - start) > 1e-8 or abs(float(core.state.t) - core.t) > 1e-7:
                    raise ValueError("Simulator clock differs from the observation schedule")
                target = tracking.target_value(case["target_schedule"], start)
                obs = observation(row["trace"][-1], bank, target)
                if agent is not None:
                    action = (agent.rng.uniform(-1., 1., 2).astype(np.float32)
                              if training and agent.env_steps < config.warmup_steps
                              else agent.action(obs, explore=training))
                    control = {"target_beta_N": target, "current_observation": obs.tolist()}
                else:
                    control = tracking.controller_command(case["controller"], target,
                                                          row["trace"][-1]["scalars"]["beta_N"], integral)
                    integral = control["integral_final"]
                    action = np.full(2, 2 * control["command_fraction"] - 1, np.float32)
                if np.shape(action) != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("Invalid bounded action")
                command = bank.from_unit_actions(action)
                row["commands"].append({"from_time": start, "to_time": end, "controller": control,
                                        "action": action.tolist(), "previous_SI": bank.prev_values.tolist(),
                                        "command_SI": command.tolist()})
                row["attempted_action_windows"] += 1
                row["pending_operation"] = {"advance_to_time": end}
                save(row)
                core.apply_updates(bank.build_updates(command, core.t, WINDOW))
                ok, done, error = core.advance()
                if not ok:
                    row["outcome"] = "solver_failure"
                    row["failure"] = {"attempted_to_time": end, "error": str(error)}
                    nxt, reward, terminal = obs.copy(), tracking_reward(None, True, index), True
                else:
                    row["completed_action_windows"] += 1
                    row["trace"].append(tracking.annotated_sample(core, bank, case, end))
                    if abs(core.t - end) > 1e-8 or abs(float(core.state.t) - core.t) > 1e-7:
                        raise ValueError("Simulator clock differs from the observation schedule")
                    if done and index < WINDOWS - 1:
                        raise ValueError("Simulator ended before the fixed horizon")
                    row["outcome"] = calibration.sampled_outcome(row["trace"][-1])
                    failed = row["outcome"] != "running"
                    # Invalid state never enters replay; a terminal failure
                    # uses the prior finite observation and zero bootstrap.
                    nxt = (obs.copy() if failed else observation(row["trace"][-1], bank,
                           tracking.target_value(case["target_schedule"], end)))
                    reward = tracking_reward(row["trace"][-1]["error_beta_N"], failed, index + 1)
                    terminal = failed or index == WINDOWS - 1
                row["rewards"].append(reward)
                row["pending_operation"] = "learner_update" if training else None
                save(row)
                if training:
                    agent.remember(obs, action, reward, nxt, terminal)
                    diagnostics = agent.train_step()
                    if diagnostics is not None:
                        row["learner_updates"].append(diagnostics)
                row["pending_operation"] = None
                save(row)
                if row["outcome"] != "running":
                    break
            else:
                row["outcome"] = "complete"
    except Exception as exc:
        row["outcome"] = "exception"
        row["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        row["case_wall_seconds"] = time.perf_counter() - started
        save(tracking.finish_case(row))
        raise
    row["case_wall_seconds"] = time.perf_counter() - started
    row["policy_hash_after"] = None if agent is None else policy_hash(agent)
    row["learner_state_after"] = None if agent is None else {"updates": agent.steps,
                "environment_steps": agent.env_steps, "replay_size": len(agent.replay)}
    if not training and row["policy_hash_before"] != row["policy_hash_after"]:
        raise RuntimeError("Evaluation mutated the actor")
    if not training and row["learner_state_before"] != row["learner_state_after"]:
        raise RuntimeError("Evaluation mutated the learner/replay")
    row["return"] = math.fsum(row["rewards"])
    save(tracking.finish_case(row))
    return row


def manifest(config):
    paths = sorted((ROOT / "hfmarl").rglob("*.py")) + [Path(tracking.__file__), Path(calibration.__file__),
            Path(calibration.shared.__file__), ROOT / "scripts/run_calibrated_td3.py"]
    sources = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}
    task_jobs = jobs(config)
    contract = {"schema": 2, "study": "calibrated_sparc_td3_development", "config": asdict(config),
                "runtime": calibration.runtime_info(), "jobs": task_jobs,
                "source_sha256": {name: calibration.sha(data) for name, data in sources.items()},
                "raw_configs": {str(job["index"]): calibration.json_value(tracking.case_config(job["case"])) for job in task_jobs},
                "calibration": {"version": tracking.CALIBRATION_VERSION,
                                "report_sha256": tracking.CALIBRATION_REPORT_SHA256},
                "observation_schema": OBS_SCHEMA, "expected_budget": expected_budget(config),
                "protocol": {"target_low": tracking.LOW, "target_high": tracking.HIGH,
                             "horizon_seconds": HORIZON, "window_seconds": WINDOW,
                             "causality": "Only current target and measured state; no schedule identity or preview. Score against target at each post-action sample, including jumps.",
                             "reward": "-min((beta_error/calibration_error_scale)^2,100); terminal failure: -100*(30-completed_windows+1)-1; finite horizon is terminal, not bootstrapped.",
                             "evaluation": "Deterministic actors, no learning or gates; every startup/transition sample scored. Conventional cases executed once and reused for paired seed comparisons.",
                             "initial_evaluation": "Each seed's untouched initialized actor is archived and evaluated on the same cases before any replay insertion or optimizer step. This additional evaluation cost is separate from training and final evaluation; it never selects a policy or changes hyperparameters.",
                             "costs": "Both methods use the calibrated target range; conventional inverse mapping additionally uses prior 120-case calibration. Prior 180-case tracking informs development. All new training/evaluation separately counted.",
                             "claim": "Development only; beta_N tracking, not profile control, beam control, federation efficacy or real-reactor validation."}}
    return {"identity": calibration.sha(calibration.canonical(contract)), **contract}, sources


def summarize(records, config, status):
    comparisons, seeds = [], []
    def key(row):
        c = row["case"]
        return c["fixed_dt"], c["initial_temperature_factor"], c["target_schedule"]
    reference = {(row["case"]["controller"], key(row)): row for row in records
                 if row["job"]["phase"] == "evaluation" and row["case"]["controller"] in CONVENTIONAL}
    for seed in config.seeds:
        selected = [row for row in records if row["job"]["seed"] == seed and row["job"]["phase"] == "evaluation"]
        normal = [row for row in selected if .9 <= row["case"]["initial_temperature_factor"] <= 1.1]
        for row in selected:
            for arm in CONVENTIONAL:
                other = reference.get((arm, key(row)))
                complete = other is not None and row["tracking_metrics"]["complete_clean_15s"] and other["tracking_metrics"]["complete_clean_15s"]
                difference = (row["tracking_metrics"]["post_action_MAE"] - other["tracking_metrics"]["post_action_MAE"] if complete else None)
                comparisons.append({"seed": seed, "case": row["case"], "comparator": arm,
                                    "both_complete": complete, "td3_minus_conventional_MAE": difference,
                                    "td3_pass": row["tracking_metrics"]["primary_pass"],
                                    "conventional_pass": None if other is None else other["tracking_metrics"]["primary_pass"]})
        expected_normal = sum(.9 <= case["initial_temperature_factor"] <= 1.1 for case in evaluation_cases(config))
        seeds.append({"seed": seed, "evaluations_recorded": len(selected), "normal_evaluations_recorded": len(normal),
                      "passes_all_configured_normal_cases": bool(expected_normal and len(normal) == expected_normal and all(row["tracking_metrics"]["primary_pass"] for row in normal)),
                      "normal_pass_count": sum(row["tracking_metrics"]["primary_pass"] for row in normal),
                      "expected_normal_cases": expected_normal})
    counts = {phase: {"episodes": sum(row["job"]["phase"] == phase for row in records),
                     "attempted_windows": sum(row["attempted_action_windows"] for row in records if row["job"]["phase"] == phase),
                     "completed_windows": sum(row["completed_action_windows"] for row in records if row["job"]["phase"] == phase)}
              for phase in ("initial_evaluation", "training", "evaluation")}
    return {"status": status, "development_only": True, "counts": counts, "seed_results": seeds,
            "critic_updates": sum(len(row["learner_updates"]) for row in records),
            "actor_updates": sum(sum(step["actor_updated"] for step in row["learner_updates"]) for row in records),
            "failures": [{"job": row["job"], "outcome": row["outcome"], "failure": row["failure"]} for row in records if row["outcome"] != "complete"],
            "paired_comparisons": comparisons,
            "timestep_comparisons": timestep_comparisons(records, config),
            "initial_timestep_comparisons": timestep_comparisons(records, config, "initial_evaluation"),
            "initial_to_final": policy_drift(records),
            "limits": "Seed-level development summaries. Conventional cases are reused, not independent seed replicates. Partial trajectories never count as passing; no shots-to-competence or general reliability claim."}


def policy_drift(records):
    def key(row):
        c = row["case"]
        return row["job"]["seed"], c["fixed_dt"], c["initial_temperature_factor"], c["target_schedule"]
    initial = {key(row): row for row in records if row["job"]["phase"] == "initial_evaluation"}
    result = []
    for row in records:
        if row["job"]["phase"] != "evaluation" or row["job"]["controller"] != "td3":
            continue
        before = initial.get(key(row))
        complete = (before is not None and before["tracking_metrics"]["complete_clean_15s"]
                    and row["tracking_metrics"]["complete_clean_15s"])
        result.append({"seed": row["job"]["seed"], "case": row["case"], "both_complete": complete,
                       "initial_pass": None if before is None else before["tracking_metrics"]["primary_pass"],
                       "final_pass": row["tracking_metrics"]["primary_pass"],
                       "final_minus_initial_MAE": row["tracking_metrics"]["post_action_MAE"] - before["tracking_metrics"]["post_action_MAE"] if complete else None})
    return result


def timestep_comparisons(records, config, phase="evaluation"):
    """Numerical screen and threshold-label changes are distinct outcomes."""
    if set(config.evaluation_dts) != {.1, .05}:
        return {"status": "not_requested", "pairs": []}
    lookup = {(row["case"]["controller"], row["job"]["seed"],
               row["case"]["initial_temperature_factor"], row["case"]["target_schedule"],
               row["case"]["fixed_dt"]): row for row in records if row["job"]["phase"] == phase}
    pairs = []
    arms = [("td3", seed) for seed in config.seeds]
    if phase == "evaluation":
        arms = [(arm, None) for arm in CONVENTIONAL] + arms
    for arm, seed in arms:
        for factor in config.evaluation_factors:
            for schedule in config.schedules:
                coarse, fine = (lookup.get((arm, seed, factor, schedule, dt)) for dt in (.1, .05))
                complete = (coarse is not None and fine is not None
                            and coarse["tracking_metrics"]["complete_clean_15s"]
                            and fine["tracking_metrics"]["complete_clean_15s"])
                exceeded, label_changed = None, None
                if complete:
                    exceeded = any(abs(a["scalars"][field] - b["scalars"][field]) > max(.001, .01 * abs(b["scalars"][field]))
                                   for a, b in zip(coarse["trace"], fine["trace"]) for field in calibration.SCREEN_FIELDS)
                    label_changed = coarse["tracking_metrics"]["primary_pass"] != fine["tracking_metrics"]["primary_pass"]
                pairs.append({"controller": arm, "seed": seed, "initial_temperature_factor": factor,
                              "target_schedule": schedule, "complete_pair": bool(complete),
                              "physical_screen_exceeded": exceeded, "tracking_pass_label_changed": label_changed,
                              "requires_review": not complete or exceeded or label_changed})
    return {"status": "complete" if all(pair["complete_pair"] for pair in pairs) else "incomplete", "pairs": pairs}


def run_study(output, config=None, resume=False):
    """Run a frozen study; resume only committed boundaries, never uncertain work."""
    config = BaselineConfig() if config is None else config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frozen, sources = manifest(config)
    if frozen["runtime"]["packages"]["torax"] != "1.4.3":
        raise ValueError("This calibrated experiment requires TORAX 1.4.3")
    state_path, manifest_path = output / "state.json", output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Output already contains a study; use a fresh directory or --resume")
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != frozen["identity"]:
            raise ValueError("Resume requires identical source, configuration and runtime")
        state = json.loads(state_path.read_text())
        if state["pending_job"] is not None:
            raise RuntimeError("Uncertain interrupted attempt preserved; refusing automatic replay")
    else:
        if resume:
            raise FileNotFoundError("No existing study to resume")
        calibration.atomic_json(manifest_path, frozen)
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for name, data in sources.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        provenance = {"source_archive_sha256": calibration.sha((output / "source.tar.gz").read_bytes())}
        try:
            provenance["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            provenance["git_head"] = None
        calibration.atomic_json(output / "provenance.json", provenance)
        state = {"identity": frozen["identity"], "next_job": 0, "pending_job": None,
                 "status": "running", "checkpoints": {}}
        state["initial_policy_checkpoints"] = {}
        calibration.atomic_json(state_path, state)
    (output / "cases").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "initial-policies").mkdir(exist_ok=True)
    records = [json.loads((output / "cases" / f"{i:05d}.json").read_text()) for i in range(state["next_job"])]
    agent, loaded_seed = None, None
    torch.set_num_threads(1)
    try:
        for job in frozen["jobs"][state["next_job"]:]:
            seed = job["seed"]
            checkpoint = output / "checkpoints" / f"seed-{seed}.pt"
            if seed is not None and loaded_seed != seed:
                agent = new_agent(config, seed)
                completed_training = sum(row["job"]["seed"] == seed and row["job"]["phase"] == "training" for row in records)
                saved = state.get("checkpoints", {}).get(str(seed))
                if saved is not None:
                    if not checkpoint.exists():
                        raise ValueError("Committed seed checkpoint is missing; refusing learner reset")
                    if saved["seed"] != seed or saved["completed_training_episodes"] != completed_training:
                        raise ValueError("Checkpoint state does not match committed training")
                    load_checkpoint(checkpoint, agent, frozen["identity"], seed, completed_training, saved["sha256"])
                else:
                    if completed_training or checkpoint.exists():
                        raise ValueError("Unbound or missing checkpoint; refusing learner reset")
                    state["checkpoints"][str(seed)] = save_checkpoint(checkpoint, agent, frozen["identity"], seed, 0)
                    initial_path = output / "initial-policies" / f"seed-{seed}.pt"
                    if initial_path.exists():
                        raise ValueError("Initial policy archive already exists; refusing overwrite")
                    initial_path.write_bytes(checkpoint.read_bytes())
                    state["initial_policy_checkpoints"][str(seed)] = dict(state["checkpoints"][str(seed)])
                    calibration.atomic_json(state_path, state)
                loaded_seed = seed
            state.update(status="running", pending_job=job["index"])
            calibration.atomic_json(state_path, state)
            row_path = output / "cases" / f"{job['index']:05d}.json"
            row = run_episode(job, config, lambda row, path=row_path: calibration.atomic_json(path, calibration.json_value(row)), agent if seed is not None else None)
            if seed is not None and job["phase"] == "training":
                state["checkpoints"][str(seed)] = save_checkpoint(checkpoint, agent, frozen["identity"], seed, job["episode"] + 1)
            records.append(row)
            state.update(next_job=job["index"] + 1, pending_job=None)
            calibration.atomic_json(state_path, state)
            calibration.atomic_json(output / "summary.json", summarize(records, config, "running"))
            print(f"{state['next_job']}/{len(frozen['jobs'])} {job['controller']} seed={seed} {job['phase']} {row['outcome']}", flush=True)
        state["status"] = "complete"
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        calibration.atomic_json(state_path, state)
        report = summarize(records, config, state["status"])
        report["pending_job"] = state["pending_job"]
        report["budget_scope"] = "Committed cases only; inspect pending case separately for attempted/uncommitted work. No resume with an uncertain pending attempt."
        calibration.atomic_json(output / "summary.json", report)
    return report
