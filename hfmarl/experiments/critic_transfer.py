"""Resumable TORAX cold-start pilot: scratch versus uniform critic handover.

This first stage is one-time multi-source representation transfer. Repeated
federation and asynchronous scheduling are deliberately separate experiments.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import time
import numpy as np
import torch

from hfmarl.agents.thermal_td3 import TD3, TD3Config, ThermalAdapter, acceptance, channel_hash
from hfmarl.devices.registry import get
from hfmarl.envs.torax_env import ToraxDeviceEnv
from hfmarl.federation.representation import UniformRepresentationServer, model_hash


def configuration(mode):
    smoke = mode == "smoke"
    cfg = TD3Config(batch=32 if smoke else 256, head_steps=4 if smoke else 100,
                    rep_steps=4 if smoke else 100, repair_steps=4 if smoke else 100,
                    adapt_steps=4 if smoke else 100)
    return {"mode": mode, "learner": asdict(cfg), "task": "moderate",
            "sources": ["iter_like", "diiid_like"],
            "targets": ["sparc_like", "tcv_like"],
            "seeds": [17] if smoke else [17, 29],
            "source_shots": 4 if smoke else 40,
            "warmup_shots": 2 if smoke else 10,
            "target_shots": 4 if smoke else 60,
            "transfer_shot": 2 if smoke else 10,
            "gate_episodes": 4 if smoke else 32,
            "evaluation_episodes": 2 if smoke else 32,
            "gate_cost_scale": "max(mean_target_warmup_training_cost,0.01); frozen_before_fork",
            "gate_tolerance": .05,
            "representation_seed": 1729,
            "initial_actor": "tanh_bias_minus1_small_output_weights",
            "warmup_actions": "uniform[-1,-0.5]; 0_to_25_percent_heater_power",
            "sharing": "one_time_equal_source_critic_trunk_delta",
            "terminal_semantics": "finite_15s_task: horizon_and_hard_events_terminal",
            "reward": "existing_env_tracking_and_limit_cost_divided_by_50; no_energy_term",
            "evidence": "wiring_only" if smoke else "development_pilot_not_confirmation",
            "adapter": ThermalAdapter.schema}


def seed_for(seed, stream, index=0):
    return int(np.random.SeedSequence([seed, stream, index]).generate_state(1)[0])


def atomic_pickle(path, state):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def code_digest():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for p in sorted((root / "hfmarl").rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class Study:
    def __init__(self, config, output, resume=False, max_operations=None):
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.output / "checkpoint.pkl"
        self.envs = {}
        self.max_operations = max_operations
        self.operations = 0
        manifest = {"experiment": config, "code_sha256": code_digest(),
                    "python": platform.python_version(),
                    "packages": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "jax", "torax"]}}
        self.identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        manifest_path = self.output / "manifest.json"
        if resume:
            # Only load a checkpoint produced by this program in a trusted output directory.
            with self.checkpoint.open("rb") as f:
                self.state = pickle.load(f)
            if self.state["identity"] != self.identity:
                raise ValueError("Resume requires identical code, config and dependency versions")
        else:
            if self.checkpoint.exists() or manifest_path.exists():
                raise ValueError("Output exists; use --resume or a fresh directory")
            self.state = {"identity": self.identity, "done": set(), "learners": {},
                          "servers": {}, "records": [], "status": "running"}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def env(self, name):
        if name not in self.envs:
            env = ToraxDeviceEnv(get(name), task=self.config["task"], seed=0)
            self.envs[name] = (env, ThermalAdapter(env))
        return self.envs[name]

    def new_learner(self, name, seed):
        env, adapter = self.env(name)
        return TD3(19, adapter.lo, adapter.hi, seed, TD3Config(**self.config["learner"]),
                   self.config["representation_seed"])

    def shot(self, name, learner, seed, training=False, random_actions=False):
        env, adapter = self.env(name)
        env.rng = np.random.default_rng(seed)
        raw, _ = env.reset()
        obs = adapter.observe(env, raw)
        cost, steps, events, tracking = 0., 0, set(), []
        start = time.monotonic()
        while True:
            action = (learner.rng.uniform(-1, -.5, 2).astype(np.float32) if random_actions
                      else learner.action(obs, explore=training))
            raw, reward, terminated, truncated, info = env.step(action)
            # On solver failure a nonfinite successor is not a bootstrap input.
            try:
                nxt = adapter.observe(env, raw)
            except ValueError:
                if not terminated:
                    raise
                nxt = np.zeros_like(obs)
            if not np.isfinite(reward):
                raise FloatingPointError("Nonfinite environment reward")
            cost -= reward
            steps += 1
            events.update(info.get("violations", ()))
            tracking.append(float(abs(raw[1])) if np.isfinite(raw[1]) else None)
            if training:
                # This is a finite-horizon task with time in the observation.
                # A completed 15s shot is terminal, not an artificial training cutoff.
                learner.remember(obs, action, reward, nxt, terminated or truncated)
                learner.train_step()
            obs = nxt
            if terminated or truncated:
                break
            if steps > env.task.steps_per_shot + 1:
                raise RuntimeError("TORAX failed to end the declared horizon")
        return {"cost": cost, "events": sorted(events), "steps": steps,
                "seconds": time.monotonic()-start, "seed": seed,
                "mean_tracking_error": float(np.mean(tracking)) if all(x is not None for x in tracking) else None}

    def save(self):
        atomic_pickle(self.checkpoint, self.state)
        tmp = self.output / "metrics.tmp"
        tmp.write_text(json.dumps(self.state["records"], indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, self.output / "metrics.json")
        (self.output / "status.json").write_text(json.dumps({"status": self.state["status"],
            "completed_operations": len(self.state["done"]), "identity": self.identity}), encoding="utf-8")

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
        print(json.dumps({"operation": key, **result}, allow_nan=False), flush=True)

    def train_shot(self, key, name, seed, index, source=False):
        learners = self.state["learners"]
        if key not in learners:
            learners[key] = self.new_learner(name, seed_for(seed, 1))
        learner = learners[key]
        result = self.shot(name, learner, seed_for(seed, 100 if source else 200, index),
                           training=True, random_actions=index < self.config["warmup_shots"])
        return {"kind": "train", "device": name, "shot": index+1,
                "learner_updates": learner.steps, "replay": len(learner.replay), **result}

    def aggregate(self, seed):
        clients = self.config["sources"]
        first = self.new_learner(clients[0], seed_for(seed, 1))
        cfg = first.cfg
        schema = channel_hash(cfg)
        server = UniformRepresentationServer(first.representation(), clients, schema,
                                              {n: "thermal-config-v1" for n in clients})
        updates = [self.state["learners"][f"{seed}/source/{n}"].contribution(
            server.tensors, n, f"{seed}/{n}/round0", schema, "thermal-config-v1") for n in clients]
        server.aggregate(updates)
        self.state["servers"][seed] = server
        return {"kind": "aggregate", "sources": clients, "version": server.version,
                "model_hash": model_hash(server.tensors), "weights": [1/len(clients)]*len(clients),
                "delta_norms": [float(np.sqrt(sum(float(np.sum(v*v)) for v in u.tensors.values()))) for u in updates]}

    def handover(self, seed, name):
        key = f"{seed}/target/{name}"
        incumbent = self.state["learners"][key]
        control = incumbent.repaired_fork()
        candidate = incumbent.repaired_fork(self.state["servers"][seed].tensors)
        costs, events, rows = [[], []], [[], []], []
        for j in range(self.config["gate_episodes"]):
            physical_seed = seed_for(seed, 300, j)
            for arm, learner in enumerate([control, candidate]):
                row = self.shot(name, learner, physical_seed)
                costs[arm].append(row["cost"])
                events[arm].append(row["events"])
                rows.append({"arm": arm, **row})
        warmup_costs = [r["cost"] for r in self.state["records"]
                        if r["operation"].startswith(key+"/warmup/")]
        cost_scale = max(float(np.mean(warmup_costs)), .01)
        decision = acceptance(*costs, *events, cost_scale,
                              seed_for(seed, 301), min_episodes=self.config["gate_episodes"],
                              tolerance=self.config["gate_tolerance"])
        self.state["learners"][key+"/control"] = control
        self.state["learners"][key+"/federated"] = candidate if decision["accept"] else copy.deepcopy(control)
        # Audit-only raw candidate kept for interpreting rejection, never used for selection later.
        self.state["learners"][key+"/candidate_at_gate"] = copy.deepcopy(candidate)
        return {"kind": "handover", "device": name, "decision": decision,
                "gate_rows": rows, "gate_shots": 2*len(costs[0]), "cost_scale": cost_scale,
                "smoke_gate_only": self.config["mode"] == "smoke"}

    def evaluate(self, seed, name):
        key = f"{seed}/target/{name}"
        rows = []
        for arm in ["control", "federated"]:
            for j in range(self.config["evaluation_episodes"]):
                row = self.shot(name, self.state["learners"][key+"/"+arm], seed_for(seed, 400, j))
                rows.append({"arm": arm, **row})
        return {"kind": "evaluation", "device": name, "seed": seed, "rows": rows,
                "split": "held_out_evaluation_never_used_for_gate"}

    def run(self):
        cfg = self.config
        if set(cfg["sources"]) & set(cfg["targets"]):
            raise ValueError("Cold-start targets must be absent from source training")
        try:
            for seed in cfg["seeds"]:
                for name in cfg["sources"]:
                    key = f"{seed}/source/{name}"
                    for shot in range(cfg["source_shots"]):
                        self.operation(f"{key}/{shot}", lambda: self.train_shot(key, name, seed, shot, True))
                self.operation(f"{seed}/aggregate", lambda: self.aggregate(seed))
                for name in cfg["targets"]:
                    key = f"{seed}/target/{name}"
                    for shot in range(cfg["transfer_shot"]):
                        self.operation(f"{key}/warmup/{shot}", lambda: self.train_shot(key, name, seed, shot))
                    self.operation(f"{key}/handover", lambda: self.handover(seed, name))
                    for shot in range(cfg["transfer_shot"], cfg["target_shots"]):
                        for arm in ["control", "federated"]:
                            self.operation(f"{key}/{arm}/{shot}", lambda: self.train_shot(key+"/"+arm, name, seed, shot))
                    self.operation(f"{key}/evaluation", lambda: self.evaluate(seed, name))
            self.state["status"] = "complete"
        except StopIteration:
            self.state["status"] = "checkpointed"
        self.save()
        return self.state["status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-operations", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    study = Study(configuration(args.mode), args.output, args.resume, args.max_operations)
    status = study.run()
    print(json.dumps({"status": status, "wall_seconds": time.monotonic()-start}), flush=True)


if __name__ == "__main__":
    main()
