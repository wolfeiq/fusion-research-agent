"""Matched-dynamics, independently seeded client critic-transfer mechanism test.

Calibration uses source-only data. Five raw target arms remain visible, including
failures. This is neither a cross-reactor result nor a cold-start speed test.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from hfmarl.agents.thermal_td3 import channel_hash, delayed_actor_update_count
from hfmarl.devices.registry import DEVICES, get
from hfmarl.experiments.critic_transfer import Study, seed_for
from hfmarl.experiments.critic_stability import StabilityStudy, configuration as stability_configuration
from hfmarl.federation.critic_strategies import calibrate_thresholds, aggregate_variants
from hfmarl.federation.representation import model_hash


ARMS = ("local", "reset_only", "uniform", "clipped", "norm_matched_uniform")


def configuration(history=10, smoke=False, *, learner_variant="standard"):
    if history not in (10, 40):
        raise ValueError("History must be 10 or 40")
    base = stability_configuration("baseline", smoke, learner_variant=learner_variant)
    effective_history = (2 if history == 10 else 4) if smoke else history
    end = 8 if smoke else 120
    offsets = [1, 2, 4] if smoke else [10, 20, 40]
    return {
        "mode": "matched", "study_version": 1, "smoke": bool(smoke),
        "learner_variant": learner_variant, "learner": base["learner"], "task": "moderate",
        "clients": {"source_a": {"device": "sparc_like", "seed_index": 1},
                    "source_b": {"device": "sparc_like", "seed_index": 2},
                    "target": {"device": "sparc_like", "seed_index": 3}},
        "sources": ["source_a", "source_b"], "target": "target", "arms": list(ARMS),
        "calibration_seed": 103 if smoke else 101, "seeds": [113] if smoke else [107, 109],
        "history_condition": history, "target_history": effective_history,
        "warmup_shots": 2 if smoke else 10, "source_shots": 4 if smoke else 40,
        "target_shots": end,
        "checkpoints": sorted({effective_history, end, *(effective_history + offset for offset in offsets
                                                         if effective_history + offset <= end)}),
        "matched_post_handover_offsets": offsets,
        "development_episodes": 2 if smoke else 4, "evaluation_episodes": 2 if smoke else 16,
        "representation_seed": 1729, "adapter": base["adapter"],
        "evidence": "wiring_only" if smoke else "matched_dynamics_development_mechanism_test",
        "primary_endpoint": ("smoke_H_to_final_cost_curve; wiring_only" if smoke else
            "trapezoidal_mean_development_cost_H_to_Hplus40; paired_within_seed; uniform_must_improve_both_local_and_reset_only_with_no_increase_in_limit_violations"),
        "secondary_endpoint": "final_120_shot_persistence" if not smoke else "smoke_final_endpoint",
        "threshold_rule": "source_only_per_branch_median_of_two_delta_norms; frozen_before_development",
        "calibration_reuse": "none; each_history_job_repeats_and_counts_its_own_calibration",
        "gate_application": "none; all_five_raw_arms_preserved",
        "q_probes": False,
        "sharing": "one_time_critic_trunk_delta_only; uniform_clipped_and_norm_matched_from_same_source_deltas",
        "norm_match_scope": "mean_learned_source_delta_per_branch; not_total_target_displacement_from_local_representation",
        "initial_actor": base["initial_actor"], "warmup_actions": base["warmup_actions"],
        "reward": base["reward"], "terminal_semantics": base["terminal_semantics"],
        "tracking_quantity": "absolute_beta_N_minus_target; not_temperature_error",
        "source_client_seed": "seed_for(replicate_seed,900,client_seed_index)",
        "representation_scope": "common_trunk_initialization_fixed_at_1729; not_independent_representation_replicates",
        "scenario_scope": "matched_SPARC_like_moderate_circular_geometry; initial_temperature_variation_only",
        "cold_start_endpoint": "not_defined; no_shots_to_competence_claim",
    }


def validate_scenario(cfg):
    if cfg.get("mode") != "matched" or cfg.get("arms") != list(ARMS):
        raise ValueError("Matched study requires all five named raw arms")
    clients, sources, target = cfg["clients"], cfg["sources"], cfg["target"]
    if (not clients or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", k) for k in clients)
            or len(sources) != 2 or len(set(sources)) != 2 or target in sources
            or set(clients) != set(sources) | {target}):
        raise ValueError("Require two distinct source IDs and a separate target client")
    indices = [client["seed_index"] for client in clients.values()]
    if len(set(indices)) != len(indices) or any(type(i) is not int or i <= 0 for i in indices):
        raise ValueError("Client seed indices must be distinct positive integers")
    devices = [client["device"] for client in clients.values()]
    if any(device not in DEVICES for device in devices) or len(set(devices)) != 1:
        raise ValueError("Clients must map to the same registered physical device")
    if cfg["representation_seed"] != 1729 or cfg["learner"]["mix"] != .25:
        raise ValueError("Common representation seed and mixing coefficient are fixed")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]) or cfg["calibration_seed"] in cfg["seeds"]:
        raise ValueError("Calibration and development seeds must be distinct")
    h, end = cfg["target_history"], cfg["target_shots"]
    if not 0 < cfg["warmup_shots"] <= h < end or cfg["source_shots"] < cfg["warmup_shots"]:
        raise ValueError("Invalid warmup, source or target budgets")
    if (cfg["checkpoints"] != sorted(set(cfg["checkpoints"])) or cfg["checkpoints"][0] != h
            or cfg["checkpoints"][-1] != end):
        raise ValueError("Checkpoints must start at handover and end at the final training shot")
    if min(cfg["development_episodes"], cfg["evaluation_episodes"]) < 1 or cfg.get("q_probes"):
        raise ValueError("Positive evaluation budgets required; Q probes disabled in this matrix")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def freeze_tensors(tensors):
    result = {key: value.copy() for key, value in tensors.items()}
    for value in result.values():
        value.setflags(write=False)
    return result


class MatchedClientStudy(StabilityStudy):
    def __init__(self, config, output, resume=False, max_operations=None):
        validate_scenario(config)
        # StabilityStudy's validator requires distinct physical source devices;
        # this separate scenario intentionally has distinct clients on one plant.
        Study.__init__(self, config, output, resume, max_operations)
        self.session_start = time.monotonic()
        self.previous_wall_seconds = self.state.get("active_wall_seconds", 0.)
        if self.state["status"] != "complete":
            self.state["status"] = "running"
        self.verify_calibration()
        self._freeze_stored_tensors()

    def client_seed(self, replicate, client_id):
        return seed_for(replicate, 900, self.config["clients"][client_id]["seed_index"])

    def metadata(self, replicate, client_id, phase):
        return {"training_seed": replicate, "client_id": client_id,
                "client_seed": self.client_seed(replicate, client_id),
                "physical_device": self.config["clients"][client_id]["device"],
                "device": self.config["clients"][client_id]["device"], "phase": phase}

    def client_key(self, replicate, client_id, phase):
        role = "source" if client_id in self.config["sources"] else "target"
        return f"{phase}/{replicate}/{role}/{client_id}"

    def source_contract(self):
        cfg = self.config
        contract = {"sources": {cid: cfg["clients"][cid] for cid in cfg["sources"]},
                "physical_devices": {cid: asdict(get(cfg["clients"][cid]["device"])) for cid in cfg["sources"]},
                "learner": cfg["learner"], "task": cfg["task"], "warmup_shots": cfg["warmup_shots"],
                "source_shots": cfg["source_shots"], "representation_seed": cfg["representation_seed"],
                "adapter": cfg["adapter"], "calibration_seed": cfg["calibration_seed"]}
        # Dataclass actuator tuples become JSON arrays. Normalize before binding
        # the in-memory artifact so byte-verified resume equality is exact.
        return json.loads(json_bytes(contract))

    def revisions(self):
        return {cid: digest({"client": self.config["clients"][cid], "contract": self.source_contract()})
                for cid in self.config["sources"]}

    def train_client(self, replicate, client_id, index, phase="development", arm=None):
        source = client_id in self.config["sources"]
        if phase == "calibration" and not source:
            raise ValueError("Calibration may only train source clients")
        if source and arm is not None:
            raise ValueError("Source clients do not have target arms")
        if arm is not None and arm not in self.config["arms"]:
            raise ValueError("Unknown target arm")
        key = self.client_key(replicate, client_id, phase) + ("/" + arm if arm else "")
        seed = self.client_seed(replicate, client_id)
        device = self.config["clients"][client_id]["device"]
        if key not in self.state["learners"]:
            if arm is not None:
                raise ValueError("Raw target arms must be created by handover")
            self.state["learners"][key] = self.new_learner(device, seed_for(seed, 1))
        learner = self.state["learners"][key]
        random_actions = index < self.config["warmup_shots"]
        row = self.shot(device, learner, seed_for(seed, 100 if source else 200, index),
                        training=True, random_actions=random_actions)
        return {"kind": "train", **self.metadata(replicate, client_id, phase), "arm": arm,
                "shot": index + 1, "learner_key": key, "learner_seed": seed_for(seed, 1),
                "role": "source" if source else "target", "random_actions": random_actions,
                "learner_updates": learner.steps, "environment_steps": learner.env_steps,
                "replay": len(learner.replay), **row}

    def source_deltas(self, replicate, phase):
        clients = self.config["sources"]
        first = clients[0]
        fresh = self.new_learner(self.config["clients"][first]["device"], seed_for(self.client_seed(replicate, first), 1))
        if "common_base" not in self.state:
            self.state["common_base"] = freeze_tensors(fresh.representation())
        base = self.state["common_base"]
        if model_hash(fresh.representation()) != model_hash(base):
            raise ValueError("Common initial representation changed between clients/replicates")
        schema, revisions = channel_hash(fresh.cfg), self.revisions()
        updates = []
        for client in clients:
            learner = self.state["learners"][self.client_key(replicate, client, phase)]
            update = learner.contribution(base, client, f"{phase}/{replicate}/{client}/round0", schema, revisions[client])
            update.tensors = freeze_tensors(update.tensors)
            updates.append(update)
        return base, updates, schema, revisions

    def source_payload(self, replicate, phase, updates):
        rows = []
        for client, update in zip(self.config["sources"], updates):
            key = self.client_key(replicate, client, phase)
            learner = self.state["learners"][key]
            params = {f"actor.{k}": v.detach().numpy() for k, v in learner.actor.state_dict().items()}
            params.update({f"critics.{k}": v.detach().numpy() for k, v in learner.critics.state_dict().items()})
            rows.append({**self.metadata(replicate, client, phase), "learner_seed": seed_for(self.client_seed(replicate, client), 1),
                         "actor_critic_hash": model_hash(params), "delta_hash": model_hash(update.tensors),
                         "environment_steps": learner.env_steps, "learner_updates": learner.steps,
                         "snapshot": self.snapshot(key, learner, self.config["source_shots"], "source_end")})
        return rows

    def calibrate(self):
        replicate = self.config["calibration_seed"]
        base, updates, schema, revisions = self.source_deltas(replicate, "calibration")
        calibrated = calibrate_thresholds(base, updates, self.config["sources"], schema, revisions)
        sources = self.source_payload(replicate, "calibration", updates)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        artifact = {"version": 1, "kind": "source_only_branch_median_thresholds",
                    "calibration_seed": replicate, "source_contract_hash": digest(self.source_contract()),
                    "source_contract": self.source_contract(),
                    "code_sha256": manifest["code_sha256"], "base_hash": model_hash(base),
                    "schema": schema, "revisions": revisions, "sources": sources,
                    "thresholds": calibrated["thresholds"], "diagnostics": calibrated["diagnostics"],
                    "source_head_fit_optimizer_steps": len(updates) * self.config["learner"]["head_steps"],
                    "source_trunk_fit_optimizer_steps": len(updates) * self.config["learner"]["rep_steps"]}
        content = json_bytes(artifact)
        tmp = self.output / "calibration.tmp"
        tmp.write_bytes(content)
        os.replace(tmp, self.output / "calibration.json")
        self.state["calibration"] = {"artifact": artifact, "artifact_sha256": hashlib.sha256(content).hexdigest(),
                                     "deltas": updates, "frozen": True}
        return {"kind": "calibration", "phase": "calibration", "training_seed": replicate,
                "thresholds": calibrated["thresholds"], "artifact_sha256": self.state["calibration"]["artifact_sha256"],
                "source_contract_hash": artifact["source_contract_hash"], "sources": sources,
                "diagnostics": calibrated["diagnostics"],
                "source_head_fit_optimizer_steps": artifact["source_head_fit_optimizer_steps"],
                "source_trunk_fit_optimizer_steps": artifact["source_trunk_fit_optimizer_steps"]}

    def verify_calibration(self):
        calibration = self.state.get("calibration")
        if calibration is None:
            return
        path = self.output / "calibration.json"
        content = path.read_bytes()
        artifact = json.loads(content)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        if (hashlib.sha256(content).hexdigest() != calibration["artifact_sha256"]
                or artifact != calibration["artifact"] or not calibration["frozen"]
                or digest(artifact["source_contract"]) != artifact["source_contract_hash"]
                or artifact["source_contract_hash"] != digest(self.source_contract())
                or artifact["code_sha256"] != manifest["code_sha256"]
                or artifact["calibration_seed"] != self.config["calibration_seed"]
                or model_hash(self.state["common_base"]) != artifact["base_hash"]):
            raise ValueError("Frozen calibration artifact/hash/config mismatch")

    def _freeze_stored_tensors(self):
        if "common_base" in self.state:
            self.state["common_base"] = freeze_tensors(self.state["common_base"])
        calibration = self.state.get("calibration")
        if calibration:
            for update in calibration["deltas"]:
                update.tensors = freeze_tensors(update.tensors)
        for bundle in self.state.get("source_bundles", {}).values():
            for update in bundle["deltas"]:
                update.tensors = freeze_tensors(update.tensors)
            bundle["proposals"] = {name: freeze_tensors(tensors) for name, tensors in bundle["proposals"].items()}

    def aggregate(self, replicate):
        self.verify_calibration()
        base, updates, schema, revisions = self.source_deltas(replicate, "development")
        thresholds = self.state["calibration"]["artifact"]["thresholds"]
        result = aggregate_variants(base, updates, self.config["sources"], schema, revisions, thresholds)
        self.state.setdefault("source_bundles", {})[replicate] = {
            "deltas": updates, "proposals": {name: freeze_tensors(tensors) for name, tensors in result["proposals"].items()},
            "diagnostics": result["diagnostics"], "base_hash": model_hash(base),
            "calibration_sha256": self.state["calibration"]["artifact_sha256"]}
        return {"kind": "aggregate", "phase": "development", "training_seed": replicate,
                "source_ids": self.config["sources"], "sources": self.source_payload(replicate, "development", updates),
                "thresholds": thresholds, "calibration_sha256": self.state["calibration"]["artifact_sha256"],
                "diagnostics": result["diagnostics"],
                "source_head_fit_optimizer_steps": len(updates) * self.config["learner"]["head_steps"],
                "source_trunk_fit_optimizer_steps": len(updates) * self.config["learner"]["rep_steps"]}

    def handover(self, replicate):
        target = self.config["target"]
        key = self.client_key(replicate, target, "development")
        incumbent = self.state["learners"][key]
        proposals = {"local": None, "reset_only": self.state["common_base"],
                     **self.state["source_bundles"][replicate]["proposals"]}
        snapshots, installed_norms, adapted_norms = {}, {}, {}
        incumbent_representation = incumbent.representation()

        def displacement_norms(values):
            return {branch: float(np.sqrt(sum(float(np.square(values[name].astype(np.float64)
                         - incumbent_representation[name].astype(np.float64)).sum())
                         for name in values if name.startswith(branch + ".")))) for branch in ("q0", "q1")}

        for arm in self.config["arms"]:
            proposal = proposals[arm]
            installed = incumbent_representation if proposal is None else {
                name: (1 - incumbent.cfg.mix) * value + incumbent.cfg.mix * proposal[name]
                for name, value in incumbent_representation.items()}
            installed_norms[arm] = displacement_norms(installed)
            learner = incumbent.repaired_fork(proposals[arm])
            adapted_norms[arm] = displacement_norms(learner.representation())
            self.state["learners"][key + "/" + arm] = learner
            snapshots[arm] = self.snapshot(key + "/" + arm, learner, self.config["target_history"], "handover")
        cfg, arms = incumbent.cfg, len(self.config["arms"])
        return {"kind": "handover", **self.metadata(replicate, target, "development"),
                "shot": self.config["target_history"], "raw_arms_preserved": True, "gate_applied": False,
                "snapshots": snapshots, "base_hash": model_hash(self.state["common_base"]),
                "proposal_hashes": {name: None if value is None else model_hash(value) for name, value in proposals.items()},
                "pre_repair_installed_displacement_norms": installed_norms,
                "post_repair_adaptation_displacement_norms": adapted_norms,
                "norm_match_scope": self.config["norm_match_scope"],
                "target_head_repair_optimizer_steps": arms * cfg.repair_steps,
                "target_adaptation_td3_updates": arms * cfg.adapt_steps,
                "target_adaptation_actor_updates": arms * delayed_actor_update_count(
                    incumbent.steps, incumbent.steps + cfg.adapt_steps,
                    cfg.policy_delay, cfg.actor_warmup_updates)}

    def evaluate_client(self, replicate, arm, shot, final=False):
        target = self.config["target"]
        key = self.client_key(replicate, target, "development") + "/" + arm
        learner = self.state["learners"][key]
        device = self.config["clients"][target]["device"]
        count = self.config["evaluation_episodes" if final else "development_episodes"]
        seed = self.client_seed(replicate, target)
        rows = [self.shot(device, learner, seed_for(seed, 600 if final else 500, i)) for i in range(count)]
        return {"kind": "evaluation", **self.metadata(replicate, target, "development"),
                "arm": arm, "shot": shot, "post_handover_shots": shot - self.config["target_history"],
                "split": "final" if final else "development", "episodes": count, "rows": rows,
                "snapshot": self.snapshot(key, learner, shot, "final" if final else "development"),
                "mean_cost": float(np.mean([r["cost"] for r in rows])),
                "violation_episodes": sum(bool(r["events"]) for r in rows),
                "learner_updates": learner.steps, "environment_steps": learner.env_steps}

    def summary(self):
        cfg, records = self.config, self.state["records"]
        training = [r for r in records if r["kind"] == "train"]
        evaluations = [r for r in records if r["kind"] == "evaluation"]
        calibration = self.state.get("calibration")
        result = {"status": self.state["status"], "identity": self.identity, "mode": "matched",
                  "history_condition": cfg["history_condition"], "target_history": cfg["target_history"],
                  "smoke": cfg["smoke"], "evidence": cfg["evidence"], "learner_variant": cfg["learner_variant"],
                  "primary_endpoint": cfg["primary_endpoint"], "clients": cfg["clients"], "seeds": cfg["seeds"],
                  "operations": len(self.state["done"]), "active_wall_seconds": self.state.get("active_wall_seconds", 0.),
                  "calibration": None if calibration is None else {"artifact_sha256": calibration["artifact_sha256"],
                      "path": "calibration.json", "frozen": calibration["frozen"], **calibration["artifact"]},
                  "final": [], "sources": [],
                  "aggregates": [{k: r[k] for k in ("training_seed", "thresholds", "calibration_sha256", "diagnostics")}
                                 for r in records if r["kind"] == "aggregate"],
                  "handovers": [{k: r[k] for k in ("training_seed", "shot", "raw_arms_preserved", "gate_applied", "proposal_hashes",
                        "pre_repair_installed_displacement_norms", "post_repair_adaptation_displacement_norms", "norm_match_scope")}
                                for r in records if r["kind"] == "handover"]}
        result["budget"] = {
            "calibration_source_training_shots": sum(r["phase"] == "calibration" for r in training),
            "development_source_training_shots": sum(r["phase"] == "development" and r["role"] == "source" for r in training),
            "target_shared_prefix_shots": sum(r["role"] == "target" and r["arm"] is None for r in training),
            "target_post_handover_training_shots": sum(r["role"] == "target" and r["arm"] is not None for r in training),
            "training_shots": len(training), "training_transitions": sum(r["steps"] for r in training),
            "development_evaluation_shots": sum(r["episodes"] for r in evaluations if r["split"] == "development"),
            "final_evaluation_shots": sum(r["episodes"] for r in evaluations if r["split"] == "final"),
            "total_simulator_episodes": len(training) + sum(r["episodes"] for r in evaluations),
            "evaluation_transitions": sum(row["steps"] for r in evaluations for row in r["rows"]),
            "gate_shots": 0, "q_probe_shots": 0}
        fit_keys = ("source_head_fit_optimizer_steps", "source_trunk_fit_optimizer_steps",
                    "target_head_repair_optimizer_steps", "target_adaptation_td3_updates", "target_adaptation_actor_updates")
        ledger = {key: sum(r.get(key, 0) for r in records) for key in fit_keys}
        ledger.update(training_td3_critic_optimizer_steps=sum(r["optimizer"]["count"] for r in training),
                      training_td3_actor_optimizer_steps=sum(r["optimizer"]["actor_updates"] for r in training))
        ledger["total_optimizer_steps"] = sum(ledger.values())
        result["optimization_budget"] = ledger
        result["compute_seconds"] = {phase: sum(r.get("operation_compute_seconds", 0.) for r in records
                                                   if r.get("phase") == phase)
                                     for phase in ("calibration", "development")}
        stages = {}
        for row in records:
            stage = row["phase"] + "/" + row["kind"]
            if row["kind"] == "train":
                stage += "/" + row["role"]
                if row["role"] == "target":
                    stage += "/" + ("shared_prefix" if row["arm"] is None else "post_handover")
            elif row["kind"] == "evaluation":
                stage += "/" + row["split"]
            stages[stage] = stages.get(stage, 0.) + row.get("operation_compute_seconds", 0.)
        result["compute_seconds_by_stage"] = stages
        for row in evaluations:
            if row["split"] != "final":
                continue
            target_key = self.client_key(row["training_seed"], cfg["target"], "development")
            relevant = [r for r in training if r["learner_key"] in (target_key, target_key + "/" + row["arm"])]
            result["final"].append({k: row[k] for k in ("training_seed", "client_id", "physical_device", "device", "arm",
                "mean_cost", "violation_episodes", "episodes", "learner_updates", "environment_steps", "snapshot")}
                | {"training_shots": len(relevant), "training_transitions": sum(r["steps"] for r in relevant),
                   "training_violation_shots": sum(bool(r["events"]) for r in relevant)})
        for phase, replicate in [("calibration", cfg["calibration_seed"]), *[("development", seed) for seed in cfg["seeds"]]]:
            for client in cfg["sources"]:
                rows = [r for r in training if r["phase"] == phase and r["training_seed"] == replicate and r["client_id"] == client]
                if rows:
                    result["sources"].append({**self.metadata(replicate, client, phase), "shots": len(rows),
                        "violation_shots": sum(bool(r["events"]) for r in rows), "transitions": sum(r["steps"] for r in rows),
                        "learner_updates": rows[-1]["learner_updates"],
                        "mean_first_five_training_cost": float(np.mean([r["cost"] for r in rows[:5]])),
                        "mean_last_five_training_cost": float(np.mean([r["cost"] for r in rows[-5:]]))})
        result["limitations"] = ["Same SPARC-like dynamics across independently seeded clients; not cross-reactor transfer.",
            "Two development seeds, conditional on one fixed common representation initialization.",
            "Branch thresholds are the median of only two source-calibration deltas; no target tuning.",
            "Each history job repeats calibration; duplicated source work is counted.",
            "History conditions change intervention timing and remaining post-handover training, not only data quantity.",
            "Raw failing policies and sources remain included; no gate fallback or policy selection.",
            "No competence threshold or cold-start speed claim is defined."]
        return result

    def save(self):
        self.state["active_wall_seconds"] = self.previous_wall_seconds + time.monotonic() - self.session_start
        super().save()

    def operation(self, key, fn):
        def timed():
            start = time.monotonic()
            result = fn()
            result["operation_compute_seconds"] = time.monotonic() - start
            return result
        super().operation(key, timed)

    def run(self, calibration_only=False):
        cfg = self.config
        try:
            calibration = cfg["calibration_seed"]
            for client in cfg["sources"]:
                key = self.client_key(calibration, client, "calibration")
                for index in range(cfg["source_shots"]):
                    self.operation(f"{key}/{index}", lambda: self.train_client(calibration, client, index, "calibration"))
            self.operation(f"calibration/{calibration}/freeze", self.calibrate)
            if calibration_only and self.state["status"] != "complete":
                self.state["status"] = "calibration_complete"
                self.save()
                return self.state["status"]
            for replicate in cfg["seeds"]:
                for client in cfg["sources"]:
                    key = self.client_key(replicate, client, "development")
                    for index in range(cfg["source_shots"]):
                        self.operation(f"{key}/{index}", lambda: self.train_client(replicate, client, index))
                self.operation(f"development/{replicate}/aggregate", lambda: self.aggregate(replicate))
                target = cfg["target"]
                key = self.client_key(replicate, target, "development")
                for index in range(cfg["target_shots"]):
                    if index < cfg["target_history"]:
                        self.operation(f"{key}/history/{index}", lambda: self.train_client(replicate, target, index))
                    else:
                        for arm in cfg["arms"]:
                            self.operation(f"{key}/{arm}/{index}", lambda: self.train_client(replicate, target, index, arm=arm))
                    if index + 1 == cfg["target_history"]:
                        self.operation(f"{key}/handover", lambda: self.handover(replicate))
                    if index + 1 in cfg["checkpoints"]:
                        for arm in cfg["arms"]:
                            self.operation(f"{key}/evaluation/{index+1}/development/{arm}",
                                           lambda: self.evaluate_client(replicate, arm, index + 1))
                for arm in cfg["arms"]:
                    self.operation(f"{key}/evaluation/{cfg['target_shots']}/final/{arm}",
                                   lambda: self.evaluate_client(replicate, arm, cfg["target_shots"], final=True))
            self.state["status"] = "complete"
        except StopIteration:
            self.state["status"] = "checkpointed"
        self.save()
        return self.state["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, choices=[10, 40], default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--learner-variant", choices=["standard", "zero_head", "slow_actor"], default="standard")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-operations", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    study = MatchedClientStudy(configuration(args.history, args.smoke, learner_variant=args.learner_variant),
                               args.output, args.resume, args.max_operations)
    status = study.run(calibration_only=args.calibration_only)
    print(json.dumps({"status": status, "history": args.history, "summary": str(Path(args.output) / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
