"""Coupled TD3/SAC research client with raw controls and a separate gate branch.

Every arm uses the same physical seed schedule, not the same action trajectories.
The raw federation arm contributes to the research coordinator; gate decisions
never censor its results. Checkpoints are trusted local Python objects only.
"""
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np

from hfmarl.agents.beam_sac import SAC, SACConfig, channel_hash as sac_channel
from hfmarl.agents.thermal_td3 import TD3, TD3Config, acceptance, delayed_actor_update_count
from hfmarl.envs.deposition_env import OBSERVATION_SCHEMA
from hfmarl.federation.repeated import ROLES
from hfmarl.federation.representation import model_hash


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment_contract(env):
    """Bind recovery to the physical task, targets and applied action semantics."""
    payload = {
        "implementation": f"{type(env).__module__}.{type(env).__qualname__}",
        "observation_schema": OBSERVATION_SCHEMA,
        "device": asdict(env.device), "task": asdict(env.task),
        "reward": asdict(env.reward_spec),
        "target_Te_keV": None if env.target_Te_keV is None else env.target_Te_keV.tolist(),
        "calibration_mode": env.calibration_mode,
        "rho_bounds": list(env.allocator.rho_bounds),
        "torax_config": copy.deepcopy(env.core.config_dict),
        "limits": [asdict(limit) for limit in env.limits.limits],
        "limit_penalty": env.limit_penalty,
        "violation_penalty": env.violation_penalty,
        "strict_initial_state": env.strict_initial_state,
        "action_cap": env.action_cap,
    }
    return {"sha256": digest(payload), "parameters": payload}


def repair_accounting(before, after, *, head_repaired=True):
    """Actual optimizer call counts; one critic optimizer updates both Qs."""
    result = {}
    for role, old, new in zip(ROLES, (before.heating, before.beam),
                              (after.heating, after.beam)):
        adaptive = new.steps - old.steps
        actor = (delayed_actor_update_count(old.steps, new.steps, new.cfg.policy_delay,
                                            new.cfg.actor_warmup_updates)
                 if role == ROLES[0] else adaptive)
        head_steps = new.cfg.repair_steps if head_repaired else 0
        result[role] = {
            "head_sgd_updates": head_steps,
            "critic_adaptation_updates": adaptive,
            "actor_adaptation_updates": actor,
            "entropy_temperature_updates": adaptive if role == ROLES[1] else 0,
            "replay_samples": (head_steps + adaptive) * new.cfg.batch,
        }
    return result


def _json_observation(value, nonfinite, path=""):
    """Preserve numerical evidence without putting invalid NaNs into JSON."""
    if isinstance(value, np.ndarray):
        return _json_observation(value.tolist(), nonfinite, path)
    if isinstance(value, np.generic):
        return _json_observation(value.item(), nonfinite, path)
    if isinstance(value, dict):
        return {str(key): _json_observation(item, nonfinite, f"{path}.{key}")
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_observation(item, nonfinite, f"{path}[{i}]")
                for i, item in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        nonfinite.append(path)
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported observation value at {path}: {type(value).__name__}")


def physical_snapshot(env):
    """Read actual simulator state; a failed read is flagged, never fabricated."""
    core = getattr(env, "core", None)
    if core is None:
        return None
    nonfinite, errors = [], {}
    result = {"t": _json_observation(getattr(core, "t", None), nonfinite, "t")}
    for label, method_name in (("scalars", "read_scalars"), ("profiles", "read_profiles")):
        method = getattr(core, method_name, None)
        if method is None:
            errors[label] = "unavailable"
            continue
        try:
            result[label] = _json_observation(method(), nonfinite, label)
        except Exception as error:
            errors[label] = f"{type(error).__name__}: {error}"
    result["read_errors"], result["nonfinite_fields"] = errors, nonfinite
    return result


@dataclass(frozen=True)
class CoupledConfig:
    episodes_per_round: int = 4
    warmup_episodes: int = 2
    gate_episodes: int = 32
    evaluation_episodes: int = 4
    training_schedule: str = "alternating"
    gate_scale: float = 1.
    gate_tolerance: float = .05
    gate_reference: str = "incumbent"
    seed: int = 17
    run_id: str = "coupled-development"

    def __post_init__(self):
        for name in ("episodes_per_round", "warmup_episodes", "gate_episodes", "evaluation_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_schedule not in {"alternating", "simultaneous"}:
            raise ValueError("Declare alternating or simultaneous training")
        if self.gate_reference not in {"incumbent", "repaired_control"}:
            raise ValueError("Declare incumbent or repaired_control gate reference")
        if not np.isfinite(self.gate_scale) or self.gate_scale <= 0:
            raise ValueError("Gate scale must be fixed, finite and positive")
        if not np.isfinite(self.gate_tolerance) or self.gate_tolerance < 0:
            raise ValueError("Gate tolerance must be finite and nonnegative")


class LearnerPair:
    def __init__(self, seed, td3_cfg, sac_cfg, power_bounds):
        self.heating = TD3(37, np.zeros(2, np.float32),
                           np.asarray(power_bounds, np.float32), seed, td3_cfg)
        self.beam = SAC(39, seed + 1, sac_cfg)

    def representations(self):
        return {ROLES[0]: self.heating.representation(), ROLES[1]: self.beam.representation()}

    def repair(self, proposal=None):
        other = copy.copy(self)
        other.heating = self.heating.repaired_fork(None if proposal is None else proposal[ROLES[0]])
        other.beam = self.beam.repaired_fork(None if proposal is None else proposal[ROLES[1]])
        return other


def episode(env, learner, seed, *, train=False, warmup=False, active="both"):
    """SAC observes the current heating proposal; store the actual next proposal.

    The episode horizon is part of this finite task: horizon completion has zero
    bootstrap, unlike an arbitrary off-policy collection cutoff.
    """
    env.rng = np.random.default_rng(seed)
    obs, reset_info = env.reset()
    initial_nonfinite = []
    initial_info = _json_observation(reset_info, initial_nonfinite, "initial_info")
    initial_physical = physical_snapshot(env)
    joint = isinstance(learner, SAC)
    random = np.random.default_rng(np.random.SeedSequence([seed, 719]))
    def heat_action(observation):
        if warmup:
            return random.uniform(-1, 1, 2).astype(np.float32)
        return learner.heating.action(observation, explore=train and active in {"both", "heating"})

    heating = None if joint else heat_action(obs)
    cost, events, trace, updates = 0., set(), [], {"heating": 0, "beam": 0, "joint": 0}
    for _ in range(env.task.steps_per_shot):
        if joint:
            action = (random.uniform(-1, 1, 4).astype(np.float32) if warmup
                      else learner.action(obs, explore=train))
            beam_obs = None
        else:
            beam_obs = np.concatenate((obs, heating)).astype(np.float32)
            beam = (random.uniform(-1, 1, 2).astype(np.float32) if warmup else
                    learner.beam.action(beam_obs, explore=train and active in {"both", "beam"}))
            action = np.concatenate((heating, beam))
        nxt, reward, terminated, truncated, info = env.step(action)
        terminal = bool(terminated or truncated)
        if not np.isfinite(reward) or not np.isfinite(nxt).all():
            raise FloatingPointError("Invalid trajectory cannot enter replay")
        next_heating = None if joint else (
            np.zeros(2, np.float32) if terminal else heat_action(nxt))
        if train:
            if joint:
                learner.remember(obs, action, reward, nxt, terminal)
                updates["joint"] += learner.train_step() is not None
            else:
                learner.heating.remember(obs, heating, reward, nxt, terminal)
                learner.beam.remember(beam_obs, beam, reward,
                    np.concatenate((nxt, next_heating)).astype(np.float32), terminal)
                if active in {"both", "heating"}:
                    updates["heating"] += learner.heating.train_step() is not None
                if active in {"both", "beam"}:
                    updates["beam"] += learner.beam.train_step() is not None
        cost -= float(reward)
        events.update(info.get("violations", ()))
        trace.append({"action": action.tolist(), "reward": float(reward),
                      "components": info.get("reward_components", {}),
                      "allocator": info.get("allocator", {}),
                      "violations": list(info.get("violations", ())),
                      "error": info.get("error"),
                      "physical": physical_snapshot(env)})
        obs, heating = nxt, next_heating
        if terminal:
            break
    else:
        raise RuntimeError("Environment failed to terminate at its declared horizon")
    return {"seed": int(seed), "cost": cost, "events": sorted(events),
            "complete": bool(truncated and not terminated), "windows": len(trace),
            "initial_info": initial_info, "initial_nonfinite_fields": initial_nonfinite,
            "initial_physical": initial_physical,
            "optimizer_updates": updates, "trace": trace}


class CoupledBackend:
    """Persistent per-facility experiment; no local learner is sent to Flower."""
    def __init__(self, source, env, directory, config, td3_cfg, sac_cfg, *, revision):
        self.source, self.env, self.directory = source, env, Path(directory)
        self.config, self.revision = config, revision
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "local-trusted-checkpoint.pkl"
        self.contract = {"source": source, "config": asdict(config),
                         "td3": asdict(td3_cfg), "sac": asdict(sac_cfg), "revision": revision,
                         "environment": environment_contract(env)}
        self.schemas = {
            ROLES[0]: digest({"algorithm": "td3", "obs": OBSERVATION_SCHEMA,
                             "actions": "aux-and-EC-budget-W1e8", "config": asdict(td3_cfg)}),
            ROLES[1]: sac_channel(sac_cfg, OBSERVATION_SCHEMA + ":current-td3-proposal2"),
        }
        if self.path.exists():
            # Never load artifacts from outside this trusted experiment folder.
            with self.path.open("rb") as handle:
                state = pickle.load(handle)
            if state["contract"] != self.contract:
                raise ValueError("Cannot resume a changed client contract")
            self.state = state
        else:
            bounds = [env.allocator.aux.hi / 1e8, env.allocator.ec.hi / 1e8]
            pair = LearnerPair(config.seed, td3_cfg, sac_cfg, bounds)
            self.state = {
                "contract": self.contract, "round": 0, "fit": None, "accept": None,
                "learners": {arm: copy.deepcopy(pair) for arm in
                             ("local", "reset_only", "raw_federated", "deployed")},
                "joint": SAC(37, config.seed + 2, sac_cfg, action_dim=4),
                "records": [], "failures": [],
            }
            self._save()

    def _save(self):
        temp = self.path.with_name(self.path.name + ".tmp")
        with temp.open("wb") as handle:
            pickle.dump(self.state, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.path)

    def _evaluate(self, learner, round_id, *, gate=False):
        count = self.config.gate_episodes if gate else self.config.evaluation_episodes
        # Separate development-gate and reporting streams; neither is confirmation.
        offset = 200000 if gate else 400000
        return [episode(self.env, learner, self.config.seed + offset + round_id * 1000 + i)
                for i in range(count)]

    def _training_episode(self, learner, seed, *, warmup, active, arm, round_id, index):
        """Runner hook for durable per-attempt evidence; default behavior unchanged."""
        return episode(self.env, learner, seed, train=True, warmup=warmup, active=active)

    def _gate_decision(self, control, candidate, round_id):
        if self.env.task.disturbance_std <= 0:
            return {"accept": False, "reason": "no_independent_gate_cases",
                    "scope": "mechanics_only; deterministic repeated trajectories"}
        return acceptance(
            [r["cost"] for r in control], [r["cost"] for r in candidate],
            [r["events"] for r in control], [r["events"] for r in candidate],
            self.config.gate_scale, self.config.seed + round_id,
            min_episodes=32, tolerance=self.config.gate_tolerance,
        )

    def _additional_evaluation(self, round_id):
        return {}

    def _export_record(self, round_id):
        """Recover the human-readable evidence from committed checkpoint data."""
        record = next(row for row in self.state["records"] if row["round"] == round_id)
        text_path = self.directory / f"round-{round_id}.json"
        temporary = text_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2, allow_nan=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, text_path)

    def _record_failure(self, before, phase, round_id, error):
        self.state = before
        self.state.setdefault("failures", []).append({
            "phase": phase, "round": round_id, "source": self.source,
            "error_type": type(error).__name__, "error": str(error),
            "environment_sha256": self.contract["environment"]["sha256"],
        })
        # Preserve the triggering error if storage itself has failed. The
        # in-memory ledger remains available to the caller in that case.
        try:
            self._save()
        except OSError:
            pass

    def contribute(self, bases, round_id):
        identity = {r: model_hash(v) for r, v in bases.items()}
        cached = self.state["fit"]
        if cached is not None and cached["round"] == round_id:
            if cached["base_hashes"] != identity:
                raise ValueError("Retry changed the coordinator base")
            return copy.deepcopy(cached["envelopes"])
        if round_id != self.state["round"] + 1:
            raise ValueError("Cannot skip or repeat completed client rounds")
        before = copy.deepcopy(self.state)
        try:
            records = {}
            arms = {**self.state["learners"], "joint_sac": self.state["joint"]}
            for name, learner in arms.items():
                records[name] = []
                for i in range(self.config.episodes_per_round):
                    index = (round_id - 1) * self.config.episodes_per_round + i
                    active = ("both" if self.config.training_schedule == "simultaneous"
                              else ("heating" if index % 2 == 0 else "beam"))
                    records[name].append(self._training_episode(
                        learner, self.config.seed + index,
                        warmup=index < self.config.warmup_episodes, active=active,
                        arm=name, round_id=round_id, index=index,
                    ))
            raw = self.state["learners"]["raw_federated"]
            envelopes = {
                role: learner.contribution(bases[role], self.source,
                    f"{self.config.run_id}:{round_id}:{self.source}:{role}",
                    self.schemas[role], self.revision)
                for role, learner in zip(ROLES, (raw.heating, raw.beam))
            }
            self.state["fit"] = {"round": round_id, "base_hashes": identity,
                                 "envelopes": envelopes}
            contribution_work = {
                role: {"head_sgd_updates": learner.cfg.head_steps,
                       "trunk_sgd_updates": learner.cfg.rep_steps,
                       "replay_samples": (learner.cfg.head_steps + learner.cfg.rep_steps) * learner.cfg.batch}
                for role, learner in zip(ROLES, (raw.heating, raw.beam))
            }
            self.state["records"].append({"round": round_id, "training": records,
                                         "contribution_optimization": contribution_work})
            self._save()
            return copy.deepcopy(envelopes)
        except Exception as error:
            self._record_failure(before, "contribution", round_id, error)
            raise

    def accept(self, proposal, round_id):
        hashes = {r: model_hash(v) for r, v in proposal.items()}
        cached = self.state["accept"]
        if cached and cached["round"] == round_id:
            if hashes != cached["hashes"]:
                raise ValueError("Retry changed proposal")
            self._export_record(round_id)
            return copy.deepcopy(cached["report"])
        if not self.state["fit"] or self.state["fit"]["round"] != round_id:
            raise ValueError("Proposal must follow the locally completed contribution")
        before = copy.deepcopy(self.state)
        try:
            learners = self.state["learners"]
            learners["reset_only"] = learners["reset_only"].repair()
            extra_repairs = {}
            if "initial_base_pull" in learners:
                if "initial_bases" not in self.state:
                    raise ValueError("Initial-base control requires its frozen untrained bases")
                learners["initial_base_pull"] = learners["initial_base_pull"].repair(self.state["initial_bases"])
                extra_repairs["initial_base_pull"] = repair_accounting(
                    before["learners"]["initial_base_pull"], learners["initial_base_pull"])
            learners["raw_federated"] = learners["raw_federated"].repair(proposal)
            # Protect the deployed learner from every candidate-side change,
            # including local repair/adaptation. The legacy comparison remains
            # explicit so older protocols can be reproduced without relabelling.
            control = (learners["deployed"] if self.config.gate_reference == "incumbent"
                       else learners["deployed"].repair())
            candidate = learners["deployed"].repair(proposal)
            a, b = self._evaluate(control, round_id, gate=True), self._evaluate(candidate, round_id, gate=True)
            gate = self._gate_decision(a, b, round_id)
            learners["deployed"] = candidate if gate["accept"] else control
            evaluation = {name: self._evaluate(learner, round_id)
                          for name, learner in {**learners, "joint_sac": self.state["joint"]}.items()}
            additional = self._additional_evaluation(round_id)
            if set(additional) & set(evaluation):
                raise ValueError("Additional evaluation cannot replace a raw control arm")
            evaluation.update(additional)
            report = {"round": round_id, "gate": gate,
                      "gate_reference": self.config.gate_reference,
                      "deployed_cost": float(np.mean([r["cost"] for r in evaluation["deployed"]])),
                      "raw_costs": {name: [r["cost"] for r in runs] for name, runs in evaluation.items()},
                      "raw_events": {name: [r["events"] for r in runs] for name, runs in evaluation.items()},
                      "gate_episodes": len(a) + len(b),
                      "evaluation_episodes": sum(map(len, evaluation.values())),
                      "training_episodes": sum(map(len, self.state["records"][-1]["training"].values())),
                      "contribution_optimization": self.state["records"][-1]["contribution_optimization"],
                      "repair_optimization": {
                          **extra_repairs,
                          "reset_only": repair_accounting(before["learners"]["reset_only"], learners["reset_only"]),
                          "raw_federated": repair_accounting(before["learners"]["raw_federated"], learners["raw_federated"]),
                          "gate_control": repair_accounting(before["learners"]["deployed"], control,
                              head_repaired=self.config.gate_reference == "repaired_control"),
                          "gate_candidate": repair_accounting(before["learners"]["deployed"], candidate),
                      },
                      "evidence": "development_only; no confirmation or safety guarantee"}
            self.state["records"][-1].update(gate_control=a, gate_candidate=b, evaluation=evaluation,
                                              report=report)
            self.state["accept"] = {"round": round_id, "hashes": hashes, "report": report}
            self.state["round"] = round_id
            self._save()
        except Exception as error:
            self._record_failure(before, "acceptance", round_id, error)
            raise
        # The checkpoint is the transaction boundary. An export failure must
        # not roll back memory and re-apply learning to already committed state.
        # A retry (also after restart) regenerates JSON from that checkpoint.
        self._export_record(round_id)
        return copy.deepcopy(report)
