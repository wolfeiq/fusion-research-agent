"""Development diagnostics for local TD3 stability and three-way critic transfer.

These studies measure raw learners, not a selected or protected controller.
The three-way run is authorized even if the baseline fails: it is diagnostic,
not confirmation of cold-start efficacy or operational safety.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from hfmarl.agents.thermal_td3 import acceptance, delayed_actor_update_count
from hfmarl.devices.registry import DEVICES
from hfmarl.experiments.critic_transfer import (
    Study, atomic_pickle, configuration as pilot_configuration, seed_for,
)
from hfmarl.federation.representation import model_hash


ARMS = ("local", "reset_only", "federated")
REFERENCE = "fixed_12p5pct"


def configuration(mode, smoke=False, *, learner_variant="standard", q_probes=False):
    if mode not in {"baseline", "threeway"}:
        raise ValueError("Unknown diagnostic mode")
    if learner_variant not in {"standard", "zero_head", "slow_actor"}:
        raise ValueError("Unknown learner variant")
    cfg = pilot_configuration("smoke" if smoke else "pilot")
    if learner_variant == "zero_head":
        cfg["learner"]["critic_head_init"] = "zero"
    elif learner_variant == "slow_actor":
        cfg["learner"]["actor_lr"] = 3e-5
    cfg.update({
        "mode": mode, "smoke": bool(smoke), "study_version": 2,
        "learner_variant": learner_variant,
        "q_probes": bool(q_probes),
        "q_probe_protocol": "six frozen-policy first-action rollouts on a shared initial state; diagnostic only",
        "devices": ["iter_like", "sparc_like", "diiid_like", "tcv_like"],
        "seeds": [17] if smoke else [17, 29, 43],
        "target_shots": 4 if smoke else 120,
        "checkpoints": [2, 4] if smoke else [10, 20, 40, 60, 90, 120],
        "development_episodes": 2 if smoke else 8,
        "evidence": "wiring_only" if smoke else "development_diagnostic_not_confirmation",
        "baseline_prerequisite": "not_established; threeway_is_diagnostic_even_if_baseline_fails",
        "reference": "constant_unit_action_-0.75; 12.5_percent_each_heater; not_optimized",
        "arms": ["local", REFERENCE] if mode == "baseline" else list(ARMS),
        "sharing": "none" if mode == "baseline" else "one_time_uniform_critic_delta_with_initial_trunk_reset_control",
        "gate_application": "advisory_only; never_replaces_any_raw_arm",
        "evaluation": "deterministic_actions; no_training; fixed_development_cases; fresh_final_seed_stream",
        "scenario_scope": "moderate_task; circular_geometry; initial_temperature_variation_only",
        "cold_start_endpoint": "not_defined; no_shots_to_competence_claim",
        "diagnostics": "command_trajectory_at_most_30_steps; optimizer_summary_per_shot",
        "action_units": "unit_commands[-1,1]; heater_fraction=(command+1)/2; device_specific_endpoints",
        "tracking_quantity": "absolute_beta_N_minus_target; not_temperature_error",
        "local_arm_semantics": "baseline_continuous_TD3; threeway_local_receives_matched_head_repair_and_adaptation",
        "rng_streams": {"source_training": 100, "target_training": 200,
                        "gate": 300, "development": 500, "final": 600,
                        "exploration_reseed_from_physical_seed": 700},
    })
    return cfg


def validate_config(cfg):
    if cfg["mode"] not in {"baseline", "threeway"}:
        raise ValueError("Unknown diagnostic mode")
    for key in ("sources", "targets", "devices"):
        names = cfg[key]
        if not names or len(set(names)) != len(names) or any(n not in DEVICES for n in names):
            raise ValueError(f"Invalid or duplicate {key}")
    if len(cfg["sources"]) < 2 or set(cfg["sources"]) & set(cfg["targets"]):
        raise ValueError("At least two source devices must be disjoint from targets")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Training seeds must be nonempty and distinct")
    checkpoints = cfg["checkpoints"]
    if (checkpoints != sorted(set(checkpoints)) or not checkpoints
            or checkpoints[0] < cfg["transfer_shot"]
            or checkpoints[-1] != cfg["target_shots"]):
        raise ValueError("Checkpoints must be unique, ordered, at/after handover and include final shot")
    if not 0 < cfg["warmup_shots"] <= cfg["transfer_shot"] < cfg["target_shots"]:
        raise ValueError("Invalid warmup/handover/training budget")
    for key in ("source_shots", "gate_episodes", "development_episodes", "evaluation_episodes"):
        if cfg[key] <= 0:
            raise ValueError(f"Positive {key} required")
    if cfg["learner"]["mix"] != .25:
        raise ValueError("This matched experiment fixes the representation mixing coefficient at .25")


class FixedPower:
    """A read-only open-loop reference; deliberately has no training methods."""
    def action(self, obs, explore=False):
        if explore:
            raise ValueError("The fixed reference is evaluation only")
        return np.full(2, -.75, np.float32)


class StabilityStudy(Study):
    def __init__(self, config, output, resume=False, max_operations=None):
        validate_config(config)
        super().__init__(config, output, resume, max_operations)
        # A resumed process is active, even though its last saved boundary was
        # labelled checkpointed. Subsequent progress files must say running.
        if self.state["status"] != "complete":
            self.state["status"] = "running"

    def shot(self, name, learner, seed, training=False, random_actions=False):
        if isinstance(learner, FixedPower) and (training or random_actions):
            raise ValueError("The fixed reference cannot be trained")
        env, adapter = self.env(name)
        env.rng = np.random.default_rng(seed)
        if training:
            # A short failed episode must not shift the exploration noise in
            # every subsequent paired episode. Replay has its own RNG stream.
            learner.rng = np.random.default_rng(seed_for(seed, 700))
        raw, _ = env.reset()
        obs = adapter.observe(env, raw)
        cost, events, tracking = 0., set(), []
        actions, trace, updates = [], [], []
        start = time.monotonic()
        steps = 0
        while True:
            action = (learner.rng.uniform(-1, -.5, 2).astype(np.float32) if random_actions
                      else learner.action(obs, explore=training))
            raw, reward, terminated, truncated, info = env.step(action)
            try:
                nxt = adapter.observe(env, raw)
            except ValueError:
                if not terminated:
                    raise
                nxt = np.zeros_like(obs)
            if not np.isfinite(reward):
                raise FloatingPointError("Nonfinite environment reward")
            cost -= float(reward)
            steps += 1
            events.update(info.get("violations", ()))
            tracking.append(float(abs(raw[1])) if np.isfinite(raw[1]) else None)
            actions.append(np.asarray(action, dtype=float).tolist())
            if len(trace) < 30:
                trace.append({"step": steps, "action": actions[-1], "reward": float(reward),
                              "tracking_error": tracking[-1],
                              "events": list(info.get("violations", ()))})
            if training:
                learner.remember(obs, action, reward, nxt, terminated or truncated)
                diagnostic = learner.train_step()
                if diagnostic is not None:
                    updates.append(diagnostic if isinstance(diagnostic, dict)
                                   else {"critic_loss": float(diagnostic)})
            obs = nxt
            if terminated or truncated:
                break
            if steps > env.task.steps_per_shot + 1:
                raise RuntimeError("TORAX failed to end the declared horizon")
        action_array = np.asarray(actions)
        optimizer = {"count": len(updates),
                     "actor_updates": sum(bool(r.get("actor_updated")) for r in updates),
                     "last": updates[-1] if updates else None}
        for key in ("critic_loss", "td_abs_mean", "critic_grad_norm", "actor_grad_norm"):
            values = [float(r[key]) for r in updates if r.get(key) is not None]
            if values:
                optimizer[key] = {"mean": float(np.mean(values)), "max": float(np.max(values))}
        return {"cost": cost, "events": sorted(events), "steps": steps,
                "seconds": time.monotonic() - start, "seed": seed,
                "mean_tracking_error": float(np.mean(tracking)) if all(x is not None for x in tracking) else None,
                "action_first": actions[0], "action_min": action_array.min(0).tolist(),
                "action_max": action_array.max(0).tolist(), "trace": trace,
                "trace_truncated": steps > len(trace), "optimizer": optimizer}

    def train_shot(self, key, name, seed, index, source=False):
        result = super().train_shot(key, name, seed, index, source)
        learner = self.state["learners"][key]
        result.update({"training_seed": seed, "learner_key": key,
                       "environment_steps": learner.env_steps,
                       "role": "source" if source else "target"})
        return result

    def aggregate(self, seed):
        # Persist the exact common initialization, before applying source deltas.
        first = self.new_learner(self.config["sources"][0], seed_for(seed, 1))
        self.state.setdefault("initial_representations", {})[seed] = first.representation()
        result = super().aggregate(seed)
        result.update({"training_seed": seed,
                       "initial_hash": model_hash(self.state["initial_representations"][seed]),
                       "source_head_fit_optimizer_steps": len(self.config["sources"]) * self.config["learner"]["head_steps"],
                       "source_trunk_fit_optimizer_steps": len(self.config["sources"]) * self.config["learner"]["rep_steps"]})
        return result

    def snapshot(self, key, learner, shot, tag):
        """Write an immutable forensic policy without duplicating replay buffers."""
        path = self.output / "policies" / key / f"shot-{shot:03d}-{tag}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"identity": self.identity, "learner_key": key, "shot": shot,
                   "tag": tag, "learner_updates": learner.steps,
                   "environment_steps": learner.env_steps,
                   "actor": copy.deepcopy(learner.actor.state_dict()),
                   "critics": copy.deepcopy(learner.critics.state_dict()),
                   "target_actor": copy.deepcopy(learner.target_actor.state_dict()),
                   "target_critics": copy.deepcopy(learner.target_critics.state_dict()),
                   "config": self.config["learner"], "adapter": self.config["adapter"]}
        atomic_pickle(path, payload)
        return path.relative_to(self.output).as_posix()

    def handover(self, seed, name):
        key = f"{seed}/target/{name}"
        incumbent = self.state["learners"][key]
        proposals = {"local": None,
                     "reset_only": self.state["initial_representations"][seed],
                     "federated": self.state["servers"][seed].tensors}
        for arm, proposal in proposals.items():
            self.state["learners"][key + "/" + arm] = incumbent.repaired_fork(proposal)
        cfg = incumbent.cfg
        adaptation_actor_updates = sum(
            delayed_actor_update_count(incumbent.steps,
                                       self.state["learners"][key + "/" + arm].steps,
                                       cfg.policy_delay, cfg.actor_warmup_updates)
            for arm in ARMS)
        rows = []
        for j in range(self.config["gate_episodes"]):
            physical_seed = seed_for(seed, 300, j)
            for arm in ARMS:
                row = self.shot(name, self.state["learners"][key + "/" + arm], physical_seed)
                rows.append({"arm": arm, **row})
        warmup_costs = [r["cost"] for r in self.state["records"]
                        if r["operation"].startswith(key + "/warmup/")]
        cost_scale = max(float(np.mean(warmup_costs)), .01)
        decisions = {}
        for control, candidate in (("local", "reset_only"), ("local", "federated"),
                                   ("reset_only", "federated")):
            left = [r for r in rows if r["arm"] == control]
            right = [r for r in rows if r["arm"] == candidate]
            decisions[f"{candidate}_vs_{control}"] = acceptance(
                [r["cost"] for r in left], [r["cost"] for r in right],
                [r["events"] for r in left], [r["events"] for r in right],
                cost_scale, seed_for(seed, 301), min_episodes=self.config["gate_episodes"],
                tolerance=self.config["gate_tolerance"])
        snapshots = {arm: self.snapshot(key + "/" + arm,
                    self.state["learners"][key + "/" + arm], self.config["transfer_shot"], "gate")
                     for arm in ARMS}
        return {"kind": "handover", "device": name, "training_seed": seed,
                "decisions": decisions, "advisory_only": True, "raw_arms_preserved": True,
                "gate_rows": rows, "gate_shots": len(rows), "cost_scale": cost_scale,
                "snapshots": snapshots, "initial_hash": model_hash(proposals["reset_only"]),
                "federated_hash": model_hash(proposals["federated"]),
                "target_head_repair_optimizer_steps": len(ARMS) * cfg.repair_steps,
                "target_adaptation_td3_updates": len(ARMS) * cfg.adapt_steps,
                "target_adaptation_actor_updates": adaptation_actor_updates}

    def evaluate_arm(self, seed, name, key, arm, shot, final=False):
        learner = FixedPower() if arm == REFERENCE else self.state["learners"][key]
        count = self.config["evaluation_episodes" if final else "development_episodes"]
        # Reuse the same development cases at every checkpoint; final cases are
        # from a disjoint seed stream and are never used for model selection.
        stream = 600 if final else 500
        rows = [self.shot(name, learner, seed_for(seed, stream, j)) for j in range(count)]
        snapshot = None if arm == REFERENCE else self.snapshot(
            key, learner, shot, "final" if final else "development")
        return {"kind": "evaluation", "device": name, "training_seed": seed,
                "arm": arm, "shot": shot, "split": "final" if final else "development",
                "rows": rows, "episodes": count, "snapshot": snapshot,
                "mean_cost": float(np.mean([r["cost"] for r in rows])),
                "violation_episodes": sum(bool(r["events"]) for r in rows),
                "learner_updates": 0 if arm == REFERENCE else learner.steps,
                "environment_steps": 0 if arm == REFERENCE else learner.env_steps}

    def evaluate_checkpoint(self, seed, name, key, shot, arms, final=False):
        tag = "final" if final else "development"
        for arm in arms:
            arm_key = key if self.config["mode"] == "baseline" else key + "/" + arm
            operation_key = f"{key}/evaluation/{shot}/{tag}/{arm}"
            self.operation(operation_key, lambda: self.evaluate_arm(
                seed, name, arm_key, arm, shot, final))
            if self.config.get("q_probes") and not final and arm != REFERENCE:
                self.operation(f"{key}/q_probe/{shot}/{arm}", lambda: self.probe_arm(
                    seed, name, arm_key, arm, shot))

    def probe_arm(self, seed, name, key, arm, shot):
        from hfmarl.experiments.critic_probes import evaluate_q_probes
        rows = evaluate_q_probes(self, name, self.state["learners"][key], seed_for(seed, 800))
        return {"kind": "q_probe", "device": name, "training_seed": seed,
                "arm": arm, "shot": shot, "episodes": len(rows), "rows": rows,
                "split": "development_diagnostic_never_used_for_training_or_selection"}

    def summary(self):
        records = self.state["records"]
        evaluations = [r for r in records if r["kind"] == "evaluation"]
        training = [r for r in records if r["kind"] == "train"]
        result = {"status": self.state["status"], "identity": self.identity,
                  "mode": self.config["mode"], "evidence": self.config["evidence"],
                  "learner_variant": self.config.get("learner_variant", "standard"),
                  "baseline_prerequisite": self.config["baseline_prerequisite"],
                  "operations": len(self.state["done"]), "final": [], "sources": [],
                  "budget": {"training_shots": len(training),
                             "training_transitions": sum(r["steps"] for r in training),
                             "development_evaluation_shots": sum(r["episodes"] for r in evaluations if r["split"] == "development"),
                             "final_evaluation_shots": sum(r["episodes"] for r in evaluations if r["split"] == "final"),
                             "q_probe_shots": sum(r["episodes"] for r in records if r["kind"] == "q_probe"),
                             "gate_shots": sum(r["gate_shots"] for r in records if r["kind"] == "handover")},
                  "gates": [{k: r[k] for k in ("device", "training_seed", "decisions", "advisory_only")}
                            for r in records if r["kind"] == "handover"],
                  "limitations": ["Only three training seeds in the full development study.",
                                  "Only initial temperature varies; no fault/regime generalization test.",
                                  "Fixed-power reference is not optimized and does not establish task achievability.",
                                  "Equal shot budgets do not imply equal transition or gradient-update budgets.",
                                  "One-time critic transfer in one process; not asynchronous or deployed federation.",
                                  "No preregistered competence endpoint or cold-start speed claim."]}
        for row in evaluations:
            if row["split"] != "final":
                continue
            seed, name, arm = row["training_seed"], row["device"], row["arm"]
            key = f"{seed}/target/{name}"
            relevant = [] if arm == REFERENCE else [r for r in training
                if r["training_seed"] == seed and r["device"] == name and r["role"] == "target"
                and (r["learner_key"] == key or r["learner_key"] == key + "/" + arm)]
            result["final"].append({k: row[k] for k in ("device", "training_seed", "arm", "mean_cost",
                "violation_episodes", "episodes", "learner_updates", "environment_steps", "snapshot")}
                | {"training_shots": len(relevant), "training_transitions": sum(r["steps"] for r in relevant),
                   "training_violation_shots": sum(bool(r["events"]) for r in relevant)})
        for seed in self.config["seeds"]:
            for name in self.config["sources"]:
                rows = [r for r in training if r["role"] == "source" and r["training_seed"] == seed and r["device"] == name]
                if rows:
                    result["sources"].append({"device": name, "training_seed": seed,
                        "shots": len(rows), "violation_shots": sum(bool(r["events"]) for r in rows),
                        "transitions": sum(r["steps"] for r in rows), "learner_updates": rows[-1]["learner_updates"]})
        training_critic_steps = sum(r["optimizer"]["count"] for r in training)
        training_actor_steps = sum(r["optimizer"]["actor_updates"] for r in training)
        fit_keys = ("source_head_fit_optimizer_steps", "source_trunk_fit_optimizer_steps",
                    "target_head_repair_optimizer_steps", "target_adaptation_td3_updates",
                    "target_adaptation_actor_updates")
        ledger = {key: sum(r.get(key, 0) for r in records) for key in fit_keys}
        ledger.update({"training_td3_critic_optimizer_steps": training_critic_steps,
                       "training_td3_actor_optimizer_steps": training_actor_steps,
                       "total_optimizer_steps": training_critic_steps + training_actor_steps + sum(ledger.values()),
                       "counting": "One joint twin-critic optimizer step counts once; actor steps count separately. "
                                   "Adaptation TD3 updates already appear in cumulative learner_updates, "
                                   "but source fitting and head repair do not. Shared warmup computed once."})
        result["optimization_budget"] = ledger
        return result

    def save(self):
        atomic_pickle(self.checkpoint, self.state)
        tmp = self.output / "metrics.tmp"
        tmp.write_text(json.dumps(self.state["records"], separators=(",", ":"), allow_nan=False), encoding="utf-8")
        os.replace(tmp, self.output / "metrics.json")
        (self.output / "status.json").write_text(json.dumps({
            "status": self.state["status"], "completed_operations": len(self.state["done"]),
            "identity": self.identity}), encoding="utf-8")
        tmp = self.output / "summary.tmp"
        tmp.write_text(json.dumps(self.summary(), indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, self.output / "summary.json")

    def operation(self, key, fn):
        if key in self.state["done"]:
            return
        if self.max_operations is not None and self.operations >= self.max_operations:
            raise StopIteration
        result = fn()
        self.state["records"].append({"operation": key, **result})
        self.state["done"].add(key)
        self.operations += 1
        self.save()
        # Detailed per-step evidence lives in metrics.json, not thousands of
        # notebook output lines. Checkpoint persistence remains exactly-once.
        keys = ("kind", "device", "training_seed", "arm", "shot", "cost", "events", "steps",
                "learner_updates", "environment_steps", "mean_cost", "violation_episodes",
                "episodes", "split", "version", "weights", "delta_norms", "decisions", "advisory_only")
        print(json.dumps({"operation": key, **{k: result[k] for k in keys if k in result}},
                         allow_nan=False), flush=True)

    def run(self):
        cfg = self.config
        try:
            for seed in cfg["seeds"]:
                if cfg["mode"] == "threeway":
                    for name in cfg["sources"]:
                        key = f"{seed}/source/{name}"
                        for shot in range(cfg["source_shots"]):
                            self.operation(f"{key}/{shot}", lambda: self.train_shot(key, name, seed, shot, True))
                    self.operation(f"{seed}/aggregate", lambda: self.aggregate(seed))
                names = cfg["devices"] if cfg["mode"] == "baseline" else cfg["targets"]
                for name in names:
                    key = f"{seed}/target/{name}"
                    for shot in range(cfg["target_shots"]):
                        if cfg["mode"] == "threeway" and shot >= cfg["transfer_shot"]:
                            for arm in ARMS:
                                self.operation(f"{key}/{arm}/{shot}", lambda: self.train_shot(key + "/" + arm, name, seed, shot))
                        else:
                            label = "warmup" if cfg["mode"] == "threeway" else "local"
                            self.operation(f"{key}/{label}/{shot}", lambda: self.train_shot(key, name, seed, shot))
                        if cfg["mode"] == "threeway" and shot + 1 == cfg["transfer_shot"]:
                            self.operation(f"{key}/handover", lambda: self.handover(seed, name))
                        if shot + 1 in cfg["checkpoints"]:
                            self.evaluate_checkpoint(seed, name, key, shot + 1, cfg["arms"])
                    self.evaluate_checkpoint(seed, name, key, cfg["target_shots"], cfg["arms"], final=True)
            self.state["status"] = "complete"
        except StopIteration:
            self.state["status"] = "checkpointed"
        self.save()
        return self.state["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "threeway"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--learner-variant", choices=["standard", "zero_head", "slow_actor"], default="standard")
    parser.add_argument("--q-probes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-operations", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    study = StabilityStudy(configuration(args.mode, args.smoke,
        learner_variant=args.learner_variant, q_probes=args.q_probes),
        args.output, args.resume, args.max_operations)
    status = study.run()
    print(json.dumps({"status": status, "mode": args.mode, "wall_seconds": time.monotonic() - start,
                      "summary": str(Path(args.output) / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
