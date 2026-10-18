"""Bounded live-TORAX cold-start study using actual Flower client/server rounds.

Two experienced simulated facilities share critic representations with a new
controller. Actors, replay and safety decisions remain local. This is a small
development study of compatible SPARC-like regimes, not real-reactor evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from hfmarl.agents.beam_sac import SACConfig, channel_hash as sac_channel
from hfmarl.agents.thermal_td3 import TD3Config
from hfmarl.devices.registry import get
from hfmarl.envs.deposition_env import PROFILE_RHO, OBSERVATION_SCHEMA, CoupledDepositionEnv, sparc_deposition_task
from hfmarl.experiments.coupled_control import CoupledBackend, CoupledConfig, LearnerPair, digest, episode, physical_snapshot
from hfmarl.federation.repeated import ROLES, RepeatedCoordinator
from hfmarl.federation.representation import model_hash

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("local", "reset_only", "initial_base_pull", "raw_federated", "deployed", "joint_sac")
FIXED = "calibration_fixed_command"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def archive_checkpoint(source, destination, metadata_path, label):
    """Retain the first durable checkpoint for each completed protocol phase."""
    destination = Path(destination)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(Path(source).read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    entry = {"filename": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    if label in metadata and metadata[label] != entry:
        raise ValueError("Archived checkpoint changed after its phase was committed")
    metadata[label] = entry
    atomic_json(metadata_path, metadata)


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...] = (17, 29)
    strategies: tuple[str, ...] = ("uniform", "clipped")
    facility_factors: tuple[tuple[str, float], ...] = (("donor_cool", .9), ("donor_warm", 1.1), ("cold_target", 1.))
    target_source: str = "cold_target"
    rounds: int = 4
    episodes_per_round: int = 6
    donor_pretrain_episodes: int = 48
    warmup_episodes: int = 2
    gate_factors: tuple[float, ...] = (.97, 1., 1.03)
    report_factors: tuple[float, ...] = (.95, 1.05)
    gate_tolerance: float = .05
    gate_cost_floor: float = .01
    gate_reference: str = "incumbent"
    heating_critic_coordinates: str = "physical"
    clipping_initial_norm_fraction: float = .1
    maximum_wall_seconds: int = 10800
    round_timeout_seconds: int = 2400
    beta_mae_tolerance: float = .005
    beta_coverage: float = .9
    profile_rmse_tolerance_keV: float = .02

    def __post_init__(self):
        if self.gate_reference not in {"incumbent", "repaired_control"}:
            raise ValueError("Gate reference must be incumbent or repaired_control")
        if self.heating_critic_coordinates not in {"physical", "normalized"}:
            raise ValueError("Heating critic coordinates must be physical or normalized")
        for name in ("rounds", "episodes_per_round", "donor_pretrain_episodes", "warmup_episodes", "maximum_wall_seconds", "round_timeout_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.episodes_per_round < 2 or self.donor_pretrain_episodes < 2:
            raise ValueError("At least two shots are needed before critic contribution")
        if self.rounds > 8 or self.episodes_per_round > 24 or self.donor_pretrain_episodes > 96:
            raise ValueError("This development runner has an explicit bounded shot budget")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(x) is not int or x < 0 for x in self.seeds):
            raise ValueError("Unique nonnegative training seeds required")
        if not self.strategies or len(set(self.strategies)) != len(self.strategies) or not set(self.strategies) <= {"uniform", "clipped"}:
            raise ValueError("Declare uniform and/or clipped strategy once")
        names = [name for name, _ in self.facility_factors]
        if len(names) != 3 or len(set(names)) != 3 or self.target_source not in names:
            raise ValueError("Exactly two donors and one named cold target required")
        if any(not name.replace("_", "").isalnum() for name in names):
            raise ValueError("Facility names must be safe identifiers")
        if any(not np.isfinite(x) or not .8 <= x <= 1.2 for _, x in self.facility_factors):
            raise ValueError("Facility factors must stay in the declared compatible regime")
        for factors in (self.gate_factors, self.report_factors):
            if not factors or len(set(factors)) != len(factors) or any(not np.isfinite(x) or not .8 <= x <= 1.2 for x in factors):
                raise ValueError("Finite unique scenario factors required")
        if set(self.gate_factors) & set(self.report_factors):
            raise ValueError("Gate and reporting conditions must be distinct")
        for x in (self.gate_tolerance, self.gate_cost_floor, self.clipping_initial_norm_fraction,
                  self.beta_mae_tolerance, self.profile_rmse_tolerance_keV):
            if not np.isfinite(x) or x <= 0:
                raise ValueError("Tolerances and clipping fraction must be finite positive values")
        if not 0 < self.beta_coverage <= 1:
            raise ValueError("Coverage must lie in (0,1]")


def config_from_dict(value):
    value = dict(value)
    # Archived manifests predate the incumbent fallback; never relabel them.
    value.setdefault("gate_reference", "repaired_control")
    value.setdefault("heating_critic_coordinates", "physical")
    for name in ("seeds", "strategies", "gate_factors", "report_factors"):
        value[name] = tuple(value[name])
    value["facility_factors"] = tuple(tuple(row) for row in value["facility_factors"])
    return StudyConfig(**value)


def learner_configs(study=None):
    """Keep archival diagnostics physical unless a study explicitly opts in."""
    common = dict(batch=32, hidden=64, reward_scale=1., gamma=float(np.exp(-.5 / 5.)),
                  head_steps=25, rep_steps=25, repair_steps=25, adapt_steps=40)
    coordinates = "physical" if study is None else study.heating_critic_coordinates
    return (TD3Config(**common, critic_head_init="zero", actor_lr=1e-4, actor_init_fraction=(.5, .5),
                      critic_action_coordinates=coordinates),
            SACConfig(**common))


def expected_budget(config):
    clients = len(config.facility_factors)
    jobs = len(config.seeds) * len(config.strategies)
    warmstart = jobs * (clients - 1) * config.donor_pretrain_episodes * 2
    initial = jobs * clients * (len(ARMS) + 1) * len(config.report_factors)
    training = jobs * clients * config.rounds * len(ARMS) * config.episodes_per_round
    gate = jobs * clients * config.rounds * 2 * len(config.gate_factors)
    evaluation = jobs * clients * config.rounds * (len(ARMS) + 1) * len(config.report_factors)
    total = warmstart + initial + training + gate + evaluation
    return {"jobs": jobs, "clients_per_job": clients, "donor_pretraining_episodes": warmstart,
            "initial_evaluation_episodes": initial, "round_training_episodes": training,
            "gate_episodes": gate, "reporting_episodes": evaluation,
            "total_episodes": total, "maximum_action_windows": total * 30,
            "maximum_nominal_solver_steps": total * 300,
            "target_training_shots_per_arm": config.rounds * config.episodes_per_round,
            "cost_scope": "Actual source pretraining is counted once before cloning donor controls; each strategy/seed job repeats and pays it. Calibration is additional."}


def empirical_gate(control, candidate, tolerance=.05, floor=.01):
    """Finite declared-case screen, never a confidence bound or safety proof."""
    reasons, differences = [], []
    if not control or len(control) != len(candidate):
        return {"accept": False, "reason": "missing_or_unpaired_cases", "scope": "finite_development_conditions_only"}
    for index, (a, b) in enumerate(zip(control, candidate)):
        if a.get("scenario") != b.get("scenario"):
            reasons.append(f"case_{index}_condition_mismatch")
        if not a.get("complete") or not b.get("complete") or a.get("windows") != 30 or b.get("windows") != 30:
            reasons.append(f"case_{index}_incomplete")
        if a.get("events") or b.get("events"):
            reasons.append(f"case_{index}_physical_event")
        if not np.isfinite([a.get("cost", np.nan), b.get("cost", np.nan)]).all():
            reasons.append(f"case_{index}_nonfinite")
            continue
        change = (b["cost"] - a["cost"]) / max(abs(a["cost"]), floor)
        differences.append(float(change))
        if change > tolerance:
            reasons.append(f"case_{index}_cost_worse_than_tolerance")
    return {"accept": not reasons, "reason": "pass" if not reasons else ";".join(reasons),
            "relative_cost_changes": differences, "tolerance": tolerance, "cost_floor": floor,
            "scope": "All fixed development conditions must pass individually; no bootstrap, independence claim, confirmation data or reactor-safety guarantee."}


def fixed_episode(env, action, seed):
    class Fixed:
        def action(self, obs, explore=False):
            return np.asarray(action, np.float32)
    # Use the same snapshot path as learned controllers without pretending this
    # conventional command is a SAC learner or inserting anything into replay.
    env.rng = np.random.default_rng(seed)
    _, info = env.reset()
    initial, trace, cost, events = physical_snapshot(env), [], 0., set()
    for _ in range(env.task.steps_per_shot):
        _, reward, terminated, truncated, detail = env.step(Fixed().action(None))
        cost -= float(reward)
        events.update(detail.get("violations", ()))
        trace.append({"action": list(action), "reward": float(reward),
                      "components": detail.get("reward_components", {}), "allocator": detail.get("allocator", {}),
                      "violations": list(detail.get("violations", ())), "error": detail.get("error"),
                      "physical": physical_snapshot(env)})
        if terminated or truncated:
            return {"seed": seed, "cost": cost, "events": sorted(events), "complete": bool(truncated and not terminated),
                    "windows": len(trace), "initial_info": info, "initial_physical": initial,
                    "optimizer_updates": {"heating": 0, "beam": 0, "joint": 0}, "trace": trace}
    raise RuntimeError("Fixed controller exceeded declared horizon")


class ScenarioBackend(CoupledBackend):
    """Fixed-scenario protocol around the existing real coupled learner backend."""
    def __init__(self, source, directory, study, reference, job):
        self.study, self.reference, self.job = study, reference, job
        self.factor = dict(study.facility_factors)[source]
        self.envs = {}
        self.pretrain_count = 0 if source == study.target_source else study.donor_pretrain_episodes
        self._gate_calls = 0
        task = sparc_deposition_task(target_beta=reference["beta_N"])
        nominal = CoupledDepositionEnv(get("sparc_like"), task=task, target_Te_keV=reference["target_Te_keV"], config_overrides={"fixed_dt": .05})
        if hashlib.sha256(json.dumps(nominal.core.config_dict, sort_keys=True, allow_nan=False).encode()).hexdigest() != reference["config_sha256"]:
            raise ValueError("Calibration reference configuration mismatch")
        self.initial_temperature = float(nominal.core.config_dict["profile_conditions"]["T_i"][0.0][0.0])
        self.task = task
        env = self.environment(self.factor)
        td3, sac = learner_configs(study)
        config = CoupledConfig(episodes_per_round=study.episodes_per_round, warmup_episodes=study.warmup_episodes,
                               gate_episodes=len(study.gate_factors), evaluation_episodes=len(study.report_factors),
                               seed=job["seed"], run_id=job["run_id"], gate_tolerance=study.gate_tolerance,
                               gate_reference=study.gate_reference)
        super().__init__(source, env, directory, config, td3, sac, revision=job["revision"])
        self.attempt_directory = self.directory / "attempts"
        self.attempt_directory.mkdir(exist_ok=True)
        for path in self.attempt_directory.glob("*.json"):
            if json.loads(path.read_text(encoding="utf-8"))["status"] != "complete":
                raise RuntimeError(f"Uncertain or failed attempt retained; refusing automatic replay: {path.name}")

    def environment(self, factor):
        key = float(factor)
        if key not in self.envs:
            self.envs[key] = CoupledDepositionEnv(get("sparc_like"), task=self.task,
                target_Te_keV=self.reference["target_Te_keV"],
                config_overrides={"fixed_dt": .05, "initial_T_keV": self.initial_temperature * key})
        return self.envs[key]

    def _save(self):
        super()._save()
        fit = self.state.get("fit")
        if self.state.get("prepared") and fit is not None:
            round_id = fit["round"]
            phase = "accepted" if self.state["round"] == round_id else "fit"
            label = f"round-{round_id}-{phase}"
            archive_checkpoint(self.path, self.directory / f"{label}-trusted.pkl",
                               self.directory / "checkpoint-archives.json", label)

    def _execute(self, learner, factor, seed, *, phase, arm, round_id, index, train=False, warmup=False, active="both"):
        path = self.attempt_directory / f"r{round_id:03d}-{phase}-{arm}-{index:04d}.json"
        if path.exists():
            raise RuntimeError("An attempted physics case cannot be silently replayed")
        context = {"source": self.source, "round": round_id, "phase": phase, "arm": arm,
                   "index": index, "seed": seed, "initial_temperature_factor": factor,
                   "training": train, "warmup": warmup, "active": active}
        record = {"context": context, "status": "attempted", "started_unix": time.time()}
        atomic_json(path, record)
        started = time.monotonic()
        env = self.environment(factor)
        try:
            result = (fixed_episode(env, self.reference["joint_action"], seed) if learner is None
                      else episode(env, learner, seed, train=train, warmup=warmup, active=active))
            result["scenario"] = {"initial_temperature_factor": factor, "phase": "gate" if phase.startswith("gate") else phase}
            record.update(status="complete", result=result)
            return result
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}",
                          last_physical_state=physical_snapshot(env),
                          work_scope="Attempt was not committed; consumed updates/windows may be incomplete. It cannot automatically replay.")
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - started
            atomic_json(path, record)

    def prepare(self):
        """Warm donors once, then preserve a pre-target-learning reference."""
        if self.state.get("prepared"):
            archive = self.directory / "initial-trusted-checkpoint.pkl"
            if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != self.state.get("initial_archive_sha256"):
                raise RuntimeError("Initial learner archive is missing or changed; refusing silent evidence loss")
            atomic_json(self.directory / "initial-evaluation.json", self.state["initial_evaluation"])
            atomic_json(self.directory / "pretraining.json", self.state["pretraining"])
            return
        if self.state["round"] or self.state["fit"] is not None:
            raise RuntimeError("Missing prepared-state contract")
        pair = self.state["learners"]["local"]
        self.state["initial_bases"] = pair.representations()
        pretraining = {"coupled_pair": [], "joint_sac": []}
        for arm, learner in (("coupled_pair", pair), ("joint_sac", self.state["joint"])):
            for index in range(self.pretrain_count):
                pretraining[arm].append(self._execute(learner, self.factor, self.config.seed + index,
                    phase="pretrain", arm=arm, round_id=0, index=index, train=True,
                    warmup=index < self.study.warmup_episodes, active="heating" if index % 2 == 0 else "beam"))
        self.state["learners"] = {arm: copy.deepcopy(pair) for arm in ARMS if arm != "joint_sac"}
        self.state["pretraining"] = pretraining
        atomic_json(self.directory / "pretraining.json", pretraining)
        evaluation = {}
        for arm, learner in {**self.state["learners"], "joint_sac": self.state["joint"], FIXED: None}.items():
            evaluation[arm] = [self._execute(learner, self.factor * mult, self.config.seed + 400000 + index,
                phase="initial", arm=arm, round_id=0, index=index) for index, mult in enumerate(self.study.report_factors)]
        initial = {"source": self.source, "pretraining_shots_per_controller": self.pretrain_count,
                   "target_training_shots": 0, "evaluation": evaluation,
                   "episodes": sum(map(len, evaluation.values())),
                   "windows": sum(row["windows"] for rows in evaluation.values() for row in rows)}
        self.state["initial_evaluation"] = initial
        self.state["prepared"] = True
        initial_path = self.directory / "initial-trusted-checkpoint.pkl"
        if initial_path.exists():
            raise RuntimeError("Refusing to overwrite initial learner archive")
        temporary = initial_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(self.state, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, initial_path)
        self.state["initial_archive_sha256"] = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        atomic_json(self.directory / "initial-evaluation.json", initial)
        self._save()

    def _training_episode(self, learner, seed, *, warmup, active, arm, round_id, index):
        return self._execute(learner, self.factor, seed + self.pretrain_count,
            phase="training", arm=arm, round_id=round_id, index=index, train=True,
            warmup=warmup and self.pretrain_count == 0, active=active)

    def _evaluate(self, learner, round_id, *, gate=False):
        factors = self.study.gate_factors if gate else self.study.report_factors
        if gate:
            arm = "control" if self._gate_calls % 2 == 0 else "candidate"
            self._gate_calls += 1
            phase, offset = "gate_" + arm, 200000
        else:
            arm = next((name for name, value in {**self.state["learners"], "joint_sac": self.state["joint"]}.items() if value is learner), None)
            if arm is None:
                raise ValueError("Unregistered reporting learner")
            phase, offset = "report", 400000
        return [self._execute(learner, self.factor * factor, self.config.seed + offset + round_id * 1000 + index,
                    phase=phase, arm=arm, round_id=round_id, index=index) for index, factor in enumerate(factors)]

    def _gate_decision(self, control, candidate, round_id):
        return empirical_gate(control, candidate, self.study.gate_tolerance, self.study.gate_cost_floor)

    def _additional_evaluation(self, round_id):
        return {FIXED: [self._execute(None, self.factor * factor, self.config.seed + 400000 + round_id * 1000 + index,
            phase="report", arm=FIXED, round_id=round_id, index=index) for index, factor in enumerate(self.study.report_factors)]}

    def contribute(self, bases, round_id):
        self.prepare()
        return super().contribute(bases, round_id)

    def accept(self, proposal, round_id):
        self._gate_calls = 0
        return super().accept(proposal, round_id)


def initial_contract(config, seed):
    td3, sac = learner_configs(config)
    device = get("sparc_like")
    bounds = [next(a.hi for a in device.actuators if a.name == name) / 1e8 for name in ("aux_heat", "ecrh")]
    pair = LearnerPair(seed, td3, sac, bounds)
    bases = pair.representations()
    schemas = {ROLES[0]: digest({"algorithm": "td3", "obs": OBSERVATION_SCHEMA,
                                "actions": "aux-and-EC-budget-W1e8", "config": asdict(td3)}),
               ROLES[1]: sac_channel(sac, OBSERVATION_SCHEMA + ":current-td3-proposal2")}
    thresholds = {role: {branch: config.clipping_initial_norm_fraction * math.sqrt(sum(
        float(np.square(value.astype(np.float64)).sum()) for name, value in tensors.items() if name.startswith(branch + ".")))
        for branch in ("q0", "q1")} for role, tensors in bases.items()}
    return bases, schemas, thresholds


def freeze_manifest(output, reference_path, config):
    reference_bytes = Path(reference_path).read_bytes()
    reference = json.loads(reference_bytes)
    if (reference.get("device") != "sparc_like" or reference.get("sampled_violations")
            or reference.get("calibration_simulated_seconds") != 15.
            or not np.array_equal(reference.get("profile_rho"), PROFILE_RHO)):
        raise ValueError("A successful compatible SPARC calibration reference is required")
    if version("torax") != "1.4.3" or version("flwr") != "1.22.0":
        raise ValueError("Study requires TORAX1.4.3 and Flower1.22.0")
    paths = sorted((ROOT / "hfmarl").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "pyproject.toml"]
    sources = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
        archive.writestr("calibration-reference.json", reference_bytes)
    td3, sac = learner_configs(config)
    manifest = {"schema": 1, "study": "live_torax_critic_federation_cold_start_development",
        "config": asdict(config), "arms": list(ARMS), "comparator": FIXED,
        "reference": reference, "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "source_file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
        "source_archive_sha256": hashlib.sha256((output / "source.zip").read_bytes()).hexdigest(),
        "learner_configs": {"td3": asdict(td3), "sac": asdict(sac)},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in ("torax", "jax", "torch", "numpy", "flwr")}},
        "expected_budget": expected_budget(config),
        "protocol": {"primary": "cold_target raw_federated versus local, reset_only and initial_base_pull after equal target training shots; endpoint cost and checkpoint-grid competence separately",
            "initial_base_pull": "Same25%representation blend and head/target/optimizer repair as federation, but pulls toward frozen UNTRAINED common initial trunks. Separates learned-source information from initialization shrinkage.",
            "donors": "Two compatible initial-temperature regimes; pretrain one coupled pair and one joint controller, then clone pair across control arms. Source cost explicitly counted.",
            "aggregation": "Actual Flower1.22 loopback gRPC RoleClient/RoleStrategy; role-matched critic trunk deltas only; actors, heads, replay, targets and optimizers remain local",
            "heating_critic_coordinates": f"{config.heating_critic_coordinates}: TD3 actor and replay actions remain normalized; physical mode maps critic action inputs to 100MW units, normalized mode retains [-1,1] coordinates. The choice is bound into the heating role schema and client recovery contract.",
            "clipping": "Per-role/per-critic norm bound fixed at .1 times initial trunk L2 norm before any training. Report activation; identical policies are possible if clipping never activates.",
            "gate": f"Reference={config.gate_reference}. All fixed gate conditions must finish without physical events and candidate cost may worsen by at most5% of max(abs(control cost),.01). Rejection retains the reference controller. Incumbent mode leaves the pre-transfer deployed learner untouched; repaired_control reproduces the legacy extra-adaptation fallback. No statistical independence or confidence interval asserted.",
            "competence": "Every reporting condition must finish30windows without events, betaMAE<=.005, at least90%beta errors within.005, and Te profileRMSE<=.02keV. These are development design tolerances, not reactor standards.",
            "first_crossing": "Evaluate at0,6,12,18,24 target shots by default, after each transfer/repair. Report first observed and first of two consecutive passing checkpoints; unresolved cases are censored. No exact shot or zero-shot transfer claim.",
            "costs": "Training, initial evaluation, gate, reporting, donor pretraining and contribution/repair optimizers separately counted. Prior calibration cost also applies.",
            "scope": "Small deterministic development matrix, known compatible SPARC-like physics. No confirmation data, real reactor efficacy, independence or safety guarantee."}}
    manifest["identity"] = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_sources(manifest):
    for name, expected in manifest["source_file_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after freezing: {name}")


def archiving_strategy(coordinator, checkpoint):
    from hfmarl.federation.flower_runtime import RoleStrategy
    checkpoint = Path(checkpoint)
    class ArchivingStrategy(RoleStrategy):
        def _archive(self, phase):
            label = f"round-{self.coordinator.version}-{phase}"
            archive_checkpoint(self.checkpoint, checkpoint.parent / f"coordinator-{label}.npz",
                               checkpoint.parent / "coordinator-checkpoints.json", label)

        def aggregate_fit(self, server_round, results, failures):
            result = super().aggregate_fit(server_round, results, failures)
            self._archive("proposal")
            return result

        def aggregate_evaluate(self, server_round, results, failures):
            result = super().aggregate_evaluate(server_round, results, failures)
            self._archive("acknowledged")
            return result
    return ArchivingStrategy(coordinator, checkpoint)


def worker(component, address, job_path, manifest_path):
    if not address.startswith("127.0.0.1:"):
        raise ValueError("Insecure Flower transport is restricted to loopback")
    import flwr as fl
    from hfmarl.federation.flower_runtime import RoleClient
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    verify_sources(manifest)
    config = config_from_dict(manifest["config"])
    td3, sac = learner_configs(config)
    if digest(manifest["learner_configs"]) != digest({"td3": asdict(td3), "sac": asdict(sac)}):
        raise ValueError("Worker learner configurations differ from the frozen manifest")
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    bases, schemas, thresholds = initial_contract(config, job["seed"])
    torch.set_num_threads(1)
    if component == "server":
        checkpoint = job_path.parent / "coordinator.npz"
        coordinator = (RepeatedCoordinator.load(checkpoint) if checkpoint.exists() else
            RepeatedCoordinator(bases, job["cohort"], schemas, {name: job["revision"] for name in job["cohort"]},
                strategy=job["strategy"], thresholds=thresholds if job["strategy"] == "clipped" else None,
                run_id=job["run_id"]))
        remaining = config.rounds - coordinator.version + int(coordinator.pending)
        if remaining > 0:
            fl.server.start_server(server_address=address,
                config=fl.server.ServerConfig(num_rounds=remaining, round_timeout=config.round_timeout_seconds),
                strategy=archiving_strategy(coordinator, checkpoint))
    else:
        if component not in job["cohort"]:
            raise ValueError("Unregistered facility")
        backend = ScenarioBackend(component, job_path.parent / "clients" / component, config, manifest["reference"], job)
        fl.client.start_client(server_address=address, insecure=True, client=RoleClient(backend, bases).to_client(),
                               max_retries=12, max_wait_time=120)


def wait_for_flower_server(address, process, deadline, *, maximum_wait=60.):
    """Wait for a live gRPC listener before starting any training client.

    Flower 1.22's legacy grpc-bidi transport ignores its retry_invoker. Its first
    Join/receive can therefore fail immediately if a client beats server startup.
    This barrier is bounded by the study deadline and never retries learner work.
    """
    import grpc

    ready_deadline = min(deadline, time.monotonic() + maximum_wait)
    if process.poll() is not None:
        raise RuntimeError(f"Flower server exited before readiness (exit {process.poll()})")
    if ready_deadline <= time.monotonic():
        raise TimeoutError("Flower server readiness exceeded its startup or wall-clock budget")
    with grpc.insecure_channel(address) as channel:
        ready = grpc.channel_ready_future(channel)
        try:
            while True:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"Flower server exited before readiness (exit {code})")
                remaining = ready_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Flower server readiness exceeded its startup or wall-clock budget")
                try:
                    ready.result(timeout=min(.25, remaining))
                except grpc.FutureTimeoutError:
                    continue
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"Flower server exited during readiness (exit {code})")
                return
        finally:
            ready.cancel()


def run_job(directory, manifest_path, manifest, config, strategy, seed, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    bases, schemas, thresholds = initial_contract(config, seed)
    job = {"strategy": strategy, "seed": seed, "cohort": [name for name, _ in config.facility_factors],
           "facility_factors": dict(config.facility_factors), "rounds": config.rounds,
           "run_id": f"{manifest['identity']}:{strategy}:seed-{seed}", "revision": manifest["identity"],
           "schemas": schemas, "thresholds": thresholds,
           "initial_base_hashes": {role: model_hash(value) for role, value in bases.items()}}
    if job_path.exists() and json.loads(job_path.read_text(encoding="utf-8")) != job:
        raise ValueError("Cannot resume a changed job contract")
    atomic_json(job_path, job)
    completed_path = directory / "coordinator.npz"
    if completed_path.exists():
        recovery_started = time.monotonic()
        completed = RepeatedCoordinator.load(completed_path)
        if completed.version == config.rounds and not completed.pending:
            if completed.run_id != job["run_id"] or completed.strategy != strategy:
                raise ValueError("Completed coordinator belongs to another frozen job")
            for round_id, history in enumerate(completed.history, start=1):
                for source in job["cohort"]:
                    record = json.loads((directory / "clients" / source / f"round-{round_id}.json").read_text(encoding="utf-8"))
                    if record["report"] != history["acknowledgements"][source]:
                        raise ValueError("Completed round evidence differs from coordinator acknowledgement")
            summary_path = directory / "summary.json"
            previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
            if previous is not None and previous.get("job") != job:
                raise ValueError("Prior job summary belongs to a different contract")
            result = {"status": "complete", "job": job, "completed_rounds": completed.version,
                      "consumed_updates": {role: len(channel.consumed) for role, channel in completed.channels.items()},
                      "scope": "Recovered completed acknowledgement; no new physics, training or transport",
                      "recovered_from_acknowledgements": True,
                      "wall_seconds": None if previous is None else previous.get("wall_seconds"),
                      "process_exit_codes": [] if previous is None else previous.get("process_exit_codes", []),
                      "recovery_wall_seconds": time.monotonic() - recovery_started,
                      "prior_summary": previous,
                      "recovery_evidence": "Original process timing and exit evidence retained when available; absent measurements remain unknown. Completion established from durable coordinator acknowledgements and matching client records."}
            atomic_json(summary_path, result)
            return result
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{probe.getsockname()[1]}"
    processes, logs, components, lifecycle = [], [], [], []
    started = time.monotonic()
    result = {"status": "running", "job": job, "scope": "live TORAX plus actual Flower gRPC"}

    def record(event, **details):
        lifecycle.append({"event": event, "elapsed_seconds": time.monotonic() - started,
                          "unix_seconds": time.time(), **details})
        atomic_json(directory / "process-lifecycle.json", lifecycle)

    try:
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", JAX_PLATFORMS="cpu")
        for component in ("server", *job["cohort"]):
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Frozen global wall-clock budget reached; pending work retained")
            log = (directory / f"{component}.log").open("ab")
            logs.append(log)
            command = [sys.executable, "-m", "scripts.run_coupled_federation", "--component", component,
                       "--address", address, "--job", str(job_path.resolve()), "--manifest", str(manifest_path.resolve())]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
            components.append(component)
            record("started", component=component, pid=getattr(processes[-1], "pid", None))
            if component == "server":
                wait_for_flower_server(address, processes[0], started + timeout)
                record("server_ready", address=address)
        while any(process.poll() is None for process in processes):
            if time.monotonic() - started > timeout:
                raise TimeoutError("Frozen global wall-clock budget reached; pending work retained")
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Flower component failed; inspect component logs and attempt ledger")
            time.sleep(.5)
        if any(process.poll() != 0 for process in processes):
            raise RuntimeError("Flower component failed; inspect component logs and attempt ledger")
        coordinator = RepeatedCoordinator.load(directory / "coordinator.npz")
        if coordinator.version != config.rounds or coordinator.pending:
            raise RuntimeError("Not all requested rounds were acknowledged")
        result.update(status="complete", completed_rounds=coordinator.version,
                      consumed_updates={role: len(channel.consumed) for role, channel in coordinator.channels.items()})
    except BaseException as error:
        observed = {component: process.poll() for component, process in zip(components, processes)}
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      process_status_before_cleanup=observed)
        record("failure_observed", error=result["error"], process_status=observed)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        result["wall_seconds"] = time.monotonic() - started
        result["process_exit_codes"] = [process.returncode for process in processes]
        result["process_components"] = components
        if result["status"] == "failed":
            result["component_log_tails"] = {
                name: (directory / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-8000:]
                for name in components
            }
        atomic_json(directory / "summary.json", result)
        if result["status"] == "failed":
            print(json.dumps({"event": "flower_job_failed", "summary": result}), flush=True)
    return result


def run(output, reference_path, config=StudyConfig(), *, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, state_path = output / "manifest.json", output / "study_state.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Use a fresh output directory or explicitly resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if config_from_dict(manifest["config"]) != config:
            raise ValueError("Cannot change a frozen configuration on resume")
        verify_sources(manifest)
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        if resume:
            raise FileNotFoundError("No frozen study to resume")
        manifest = freeze_manifest(output, reference_path, config)
        state = {"status": "running", "identity": manifest["identity"], "completed_jobs": [], "pending_job": None, "wall_seconds": 0.}
    started, previous_wall = time.monotonic(), state["wall_seconds"]
    try:
        for strategy in config.strategies:
            for seed in config.seeds:
                name = f"{strategy}-seed-{seed}"
                if name in state["completed_jobs"]:
                    continue
                remaining = config.maximum_wall_seconds - previous_wall - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Study wall-clock budget exhausted")
                state.update(status="running", pending_job=name)
                atomic_json(state_path, state)
                run_job(output / "jobs" / name, manifest_path, manifest, config, strategy, seed, remaining)
                state["completed_jobs"].append(name)
                state.update(pending_job=None, wall_seconds=previous_wall + time.monotonic() - started)
                atomic_json(state_path, state)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["wall_seconds"] = previous_wall + time.monotonic() - started
        atomic_json(state_path, state)
        atomic_json(output / "study_summary.json", {**state, "expected_budget": expected_budget(config),
                    "interpretation": "Use independent artifact analysis for competence and paired results; completion alone does not establish transfer benefit."})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--reference")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29])
    parser.add_argument("--strategies", nargs="+", choices=("uniform", "clipped"), default=["uniform", "clipped"])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--episodes-per-round", type=int, default=6)
    parser.add_argument("--donor-pretrain-episodes", type=int, default=48)
    parser.add_argument("--maximum-wall-seconds", type=int, default=10800)
    parser.add_argument("--gate-reference", choices=("incumbent", "repaired_control"), default="incumbent")
    parser.add_argument("--heating-critic-coordinates", choices=("physical", "normalized"), default="physical")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--component")
    parser.add_argument("--address")
    parser.add_argument("--job")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.component:
        if not args.address or not args.job or not args.manifest:
            parser.error("Workers require address, job and manifest")
        worker(args.component, args.address, args.job, args.manifest)
        return
    config = StudyConfig(seeds=tuple(args.seeds), strategies=tuple(args.strategies), rounds=args.rounds, episodes_per_round=args.episodes_per_round,
                         donor_pretrain_episodes=args.donor_pretrain_episodes, maximum_wall_seconds=args.maximum_wall_seconds,
                         gate_reference=args.gate_reference, heating_critic_coordinates=args.heating_critic_coordinates)
    if args.smoke:
        config = replace(config, seeds=config.seeds[:1], rounds=1, episodes_per_round=2, donor_pretrain_episodes=2,
                         gate_factors=(1.,), report_factors=(1.01,))
    if args.budget_only:
        print(json.dumps(expected_budget(config), indent=2))
        return
    if not args.output or not args.reference:
        parser.error("Study needs output and calibration reference")
    print(json.dumps(run(args.output, args.reference, config, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
