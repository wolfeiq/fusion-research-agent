"""Bounded, isolated live-TORAX TD3/SAC transfer pilot.

The CLI always constructs TORAX. Tests may inject a synthetic environment, but
there is no mock/surrogate runtime mode. Aggregation uses the real role-matched
critic implementation in process; this does not test Flower network transport.
Only this invocation's output directory is written, never existing experiments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import threading
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

ARMS = ("local", "reset_only", "uniform", "clipped")
BASELINE = {
    "td3_actor_lr": .0001, "sac_actor_lr": .0003, "sac_alpha_lr": .0003,
    "td3_mix": .25, "sac_mix": .25, "clipping_initial_norm_fraction": .1,
    "heating_critic_coordinates": "normalized",
}


def validate_config(value):
    """Validate the complete, non-executable request without ML imports."""
    if not isinstance(value, dict) or set(value) != {"schema", "change", "value", "seeds", "rounds"}:
        raise ValueError("Expected only schema, change, value, seeds and rounds")
    if (value["schema"] != "fusion-shadow-config-v1" or not isinstance(value["change"], str)
            or value["change"] not in BASELINE):
        raise ValueError("Unknown shadow schema or change")
    change, proposed = value["change"], value["value"]
    if change == "heating_critic_coordinates":
        if proposed not in ("normalized", "physical"):
            raise ValueError("Unsupported critic coordinates")
    else:
        if type(proposed) not in (int, float) or not math.isfinite(proposed):
            raise ValueError("Candidate must be a finite scalar number")
        lo, hi = ((1e-6, 1e-3) if change.endswith("lr") else
                  (0., 1.) if change.endswith("mix") else (.001, 1.))
        if not lo <= proposed <= hi:
            raise ValueError("Candidate outside bounded policy range")
        proposed = float(proposed)
    if proposed == BASELINE[change]:
        raise ValueError("Candidate must change exactly one baseline value")
    seeds = value["seeds"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(seed) is not int or not 0 <= seed <= 1_000_000 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Require one to three unique bounded integer seeds")
    if type(value["rounds"]) is not int or not 1 <= value["rounds"] <= 2:
        raise ValueError("Require one or two rounds")
    return {"schema": value["schema"], "change": change, "value": proposed,
            "seeds": list(seeds), "rounds": value["rounds"]}


def specification(config):
    """Frozen conditions and full episode budget, shared by both configurations."""
    config = validate_config(config)
    seeds, rounds = len(config["seeds"]), config["rounds"]
    return {
        "seeds": config["seeds"], "rounds": rounds, "arms": list(ARMS),
        "device": "sparc_like", "donors": {"donor_cool": .9, "donor_warm": 1.1},
        "target_initial_temperature_factor": 1., "donor_training_shots": 4,
        "target_training_shots_per_round": 2, "warmup_shots": 1,
        "training_schedule": "simultaneous TD3 and SAC updates; 30 windows per shot",
        "gate_factor": .95, "report_factor": 1.05,
        "gate_tolerance": .05, "gate_cost_floor": .01,
        "competence": {"windows": 30, "beta_mae": .005, "beta_coverage": .9,
                       "profile_rmse_keV": .02},
        "seed_schedule": "training=seed*10000+source_index*100+shot; gate=seed*10000+8000; report=seed*10000+9000",
        "source_cost_scope": "Two donors each train four shots once per configuration and seed. Frozen donor replay supplies newly fitted common-base updates each round; donor policies do not receive the aggregate.",
        "transfer_scope": "One-way repeated transfer pilot; each strategy keeps its own common-base history and continuing raw target. Actors, replay, heads, entropy and optimizers stay local.",
        "transport_scope": "Real role-matched aggregate_variants in process; no Flower client/server or network-transport validation.",
        "gate_scope": "Per-round candidate versus pre-transfer incumbent on separate gate conditions. Deployed is a diagnostic selected report, not a separately trained longitudinal branch.",
        "metric_scope": "Endpoint cost, beta_N tracking and temperature-profile RMSE are separate. Checkpoint competence is not shots-to-competence or cold-start efficacy.",
        "budget": {"training_episodes": seeds * (16 + 16 * rounds),
                   "evaluation_episodes": seeds * (4 + 20 * rounds),
                   "total_episodes": seeds * (20 + 36 * rounds),
                   "maximum_windows": seeds * (20 + 36 * rounds) * 30,
                   "target_training_shots_per_arm": 2 * rounds,
                   "configurations": 2, "calibration_cost": "Prior calibration excluded; calibration reference retained."},
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _make_environments(reference):
    """Runtime always uses live TORAX and binds targets to calibrated physics."""
    from hfmarl.devices.registry import get
    from hfmarl.envs.deposition_env import CoupledDepositionEnv, sparc_deposition_task
    if (reference["device"] != "sparc_like" or reference["sampled_violations"]
            or reference["calibration_simulated_seconds"] != 15.
            or reference["joint_action"] != [0., 0., 0., 1.]):
        raise ValueError("Require the original clean calibrated SPARC reference")
    task = sparc_deposition_task(target_beta=reference["beta_N"])
    nominal = CoupledDepositionEnv(get("sparc_like"), task=task,
        target_Te_keV=reference["target_Te_keV"], config_overrides={"fixed_dt": .05})
    actual = hashlib.sha256(json.dumps(nominal.core.config_dict, sort_keys=True,
                                       allow_nan=False).encode()).hexdigest()
    if actual != reference["config_sha256"]:
        raise ValueError("Calibration reference configuration mismatch")
    temperature = float(nominal.core.config_dict["profile_conditions"]["T_i"][0.0][0.0])
    envs = {1.: nominal}

    def environment(factor):
        if factor not in envs:
            envs[factor] = CoupledDepositionEnv(get("sparc_like"), task=task,
                target_Te_keV=reference["target_Te_keV"], config_overrides={
                    "fixed_dt": .05, "initial_T_keV": temperature * factor})
        return envs[factor]

    return environment, [nominal.allocator.aux.hi / 1e8, nominal.allocator.ec.hi / 1e8]


def _learner_configs(settings):
    from scripts.run_coupled_federation import learner_configs
    td3, sac = learner_configs()
    return (replace(td3, actor_lr=settings["td3_actor_lr"], mix=settings["td3_mix"],
                    critic_action_coordinates=settings["heating_critic_coordinates"]),
            replace(sac, actor_lr=settings["sac_actor_lr"], alpha_lr=settings["sac_alpha_lr"],
                    mix=settings["sac_mix"]))


def run(config, reference_path, output, *, max_seconds=600):
    """Run a fresh bounded study; save partial evidence and propagate failures."""
    config = validate_config(config)
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800:
        raise ValueError("Maximum wall time must lie in (0, 1800]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty shadow output directory")
    start = time.monotonic()
    conditions = specification(config)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    conditions["reference_sha256"] = _digest(reference)
    result = {"schema": "fusion-shadow-worker-result-v1", "status": "running",
              "config": config, "conditions": conditions, "conditions_sha256": _digest(conditions),
              "reference": reference, "failures": [], "limitations": [
                  "Development pilot only; one to three seeds and one to two rounds cannot establish efficacy.",
                  "Fresh initialization, not resumption of active experiment checkpoints.",
                  "Tiny simultaneous-update schedule differs from the main alternating training study.",
                  "Raw policy continues after a gate rejection; deployed summaries are per-round diagnostics.",
                  "No magnetic-island suppression, mirror steering or real-reactor safety claim."]}
    for name in ("baseline", "candidate"):
        result[name] = {"status": "pending", "seeds": [], "accounting": {
            "training_episodes": 0, "evaluation_episodes": 0, "windows": 0,
            "training_optimizer_updates": {"heating": 0, "beam": 0, "joint": 0},
            "contribution_optimization": [], "repair_optimization": []}}
    phase = "initializing"
    current_seed = None

    def save():
        result["elapsed_seconds"] = time.monotonic() - start
        _write(output / "result.json", result)
        _write(output / "progress.json", {"status": result["status"], "phase": phase,
            "elapsed_seconds": result["elapsed_seconds"], "completed_episodes": sum(
                result[name]["accounting"][key] for name in ("baseline", "candidate")
                for key in ("training_episodes", "evaluation_episodes")),
            "maximum_episodes": conditions["budget"]["total_episodes"]})

    def check_time():
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError("Frozen shadow wall-time budget exhausted")

    save()
    try:
        import numpy as np
        import torch

        from hfmarl.agents.thermal_td3 import delayed_actor_update_count
        from hfmarl.experiments.coupled_control import (
            LearnerPair,
            episode,
            repair_accounting,
        )
        from hfmarl.federation.critic_strategies import aggregate_variants
        from hfmarl.federation.repeated import ROLES
        from hfmarl.federation.representation import model_hash
        from scripts.analyze_coupled_federation import meets_competence, summarize_cases
        from scripts.run_coupled_federation import empirical_gate, fixed_episode
        from scripts.stabilize_td3_heating import pair_state_digest

        torch.set_num_threads(1)
        result["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "packages": {name: version(name) for name in ("torch", "torax", "jax", "numpy", "flwr")}}
        environment, bounds = _make_environments(reference)
        (output / "episodes").mkdir()
        (output / "aggregation").mkdir()

        def summary(cases):
            value = summarize_cases(cases, reference)
            value["competent"] = meets_competence(value, **conditions["competence"])
            return value

        def sound(value):
            return (value["complete_cases"] == value["cases"]
                    and not value["cases_with_events"] and not value["cases_with_invalid_physics"]
                    and all(row["windows"] == 30 for row in value["per_case"]))

        for variant in ("baseline", "candidate"):
            settings = dict(BASELINE)
            if variant == "candidate":
                settings[config["change"]] = config["value"]
            td3, sac = _learner_configs(settings)
            item = result[variant]
            item.update(status="running", configs={"td3": asdict(td3), "sac": asdict(sac),
                "clipping_initial_norm_fraction": settings["clipping_initial_norm_fraction"]})
            for seed in config["seeds"]:
                check_time()
                seed_row = {"seed": seed, "status": "running", "conditions_sha256": result["conditions_sha256"],
                            "arms": {}, "rounds": [], "training_stable": True}
                current_seed = seed_row
                item["seeds"].append(seed_row)
                target = LearnerPair(seed, td3, sac, bounds)
                initial_base = target.representations()
                seed_row["initial_pair_digest"] = pair_state_digest(target)
                seed_row["initial_representation_hashes"] = {
                    role: model_hash(values) for role, values in initial_base.items()}
                schemas = {role: _digest({"role": role, "td3": asdict(td3), "sac": asdict(sac),
                                          "observation": "coupled-deposition"}) for role in ROLES}
                thresholds = {role: {branch: settings["clipping_initial_norm_fraction"] * float(np.sqrt(sum(
                    np.square(value.astype(np.float64)).sum() for key, value in initial_base[role].items()
                    if key.startswith(branch + ".")))) for branch in ("q0", "q1")} for role in ROLES}
                count = 0

                def collect(pair, factor, shot_seed, label, *, train=False, warmup=False, fixed=False,
                            variant=variant, seed=seed, item=item, seed_row=seed_row):
                    nonlocal phase, count
                    check_time()
                    phase = f"{variant}/seed-{seed}/{label}"
                    count += 1
                    path = output / "episodes" / f"{variant}-{seed}-{count:03d}.json"
                    attempt = {"status": "started", "phase": phase, "train": train,
                               "scenario": {"initial_temperature_factor": factor}, "seed": shot_seed}
                    _write(path, attempt)
                    save()
                    before = pair_state_digest(pair) if not train and not fixed else None
                    case = (fixed_episode(environment(factor), reference["joint_action"], shot_seed) if fixed
                            else episode(environment(factor), pair, shot_seed, train=train,
                                         warmup=warmup, active="both"))
                    case["scenario"] = attempt["scenario"]
                    attempt.update(status="complete", result=case)
                    _write(path, attempt)
                    account = item["accounting"]
                    account["training_episodes" if train else "evaluation_episodes"] += 1
                    account["windows"] += case["windows"]
                    for role, updates in case["optimizer_updates"].items():
                        account["training_optimizer_updates"][role] += updates
                    measured = summary([case])
                    if before is not None and pair_state_digest(pair) != before:
                        raise RuntimeError("Evaluation changed a learner's state")
                    if train and not sound(measured):
                        seed_row["training_stable"] = False
                        seed_row.setdefault("training_failures", []).append(measured)
                        save()
                        raise RuntimeError("Incomplete or invalid training trajectory; diagnose before transfer")
                    save()
                    return case

                report_seed, gate_seed = seed * 10000 + 9000, seed * 10000 + 8000
                seed_row["initial"] = summary([collect(target, 1.05, report_seed, "initial")])
                seed_row["fixed_reference"] = summary([collect(target, 1.05, report_seed, "fixed", fixed=True)])
                donors = {}
                for source_index, (source, factor) in enumerate(conditions["donors"].items(), start=1):
                    donors[source] = copy.deepcopy(target)
                    for shot in range(4):
                        collect(donors[source], factor, seed * 10000 + source_index * 100 + shot,
                                f"{source}/train-{shot}", train=True, warmup=shot == 0)
                donor_hashes = {source: pair_state_digest(pair) for source, pair in donors.items()}
                arms = {arm: copy.deepcopy(target) for arm in ARMS}
                bases = {arm: copy.deepcopy(initial_base) for arm in ("uniform", "clipped")}
                for round_id in range(1, config["rounds"] + 1):
                    round_row = {"round": round_id, "target_training_shots_per_arm": 2 * round_id,
                                 "arms": {}, "aggregation": {}}
                    seed_row["rounds"].append(round_row)
                    for arm in ARMS:
                        for shot in range(2 * (round_id - 1), 2 * round_id):
                            collect(arms[arm], 1., seed * 10000 + shot, f"r{round_id}/{arm}/train-{shot}",
                                    train=True, warmup=shot == 0)
                        check_time()
                        incumbent = arms[arm]
                        if arm == "reset_only":
                            arms[arm] = incumbent.repair()
                            item["accounting"]["repair_optimization"].append({"seed": seed, "round": round_id,
                                "arm": arm, "roles": repair_accounting(incumbent, arms[arm])})
                        if arm in ("uniform", "clipped"):
                            proposals, arrays = {}, {}
                            cohort = (*donors, "cold_target")
                            revisions = {source: "frozen-shadow-v1" for source in cohort}
                            for role, member in zip(ROLES, ("heating", "beam")):
                                base = bases[arm][role]
                                updates = []
                                for source, pair in (*donors.items(), ("cold_target", incumbent)):
                                    check_time()
                                    learner = getattr(pair, member)
                                    update = learner.contribution(base, source,
                                        f"{variant}:{seed}:{arm}:{round_id}:{source}:{role}", schemas[role], revisions[source])
                                    updates.append(update)
                                    item["accounting"]["contribution_optimization"].append({
                                        "seed": seed, "round": round_id, "arm": arm, "source": source, "role": role,
                                        "head_sgd_updates": learner.cfg.head_steps, "trunk_sgd_updates": learner.cfg.rep_steps,
                                        "frozen_replay_samples": max(learner.cfg.head_steps, learner.cfg.rep_steps) * learner.cfg.batch,
                                        "optimization_sample_uses": (learner.cfg.head_steps + learner.cfg.rep_steps) * learner.cfg.batch})
                                    for key, value in update.tensors.items():
                                        arrays[f"{role}/delta/{source}/{key}"] = value
                                aggregate = aggregate_variants(base, updates, cohort, schemas[role], revisions, thresholds[role])
                                proposals[role] = aggregate["proposals"][arm]
                                round_row["aggregation"].setdefault(arm, {})[role] = aggregate["diagnostics"]
                                for key, value in base.items():
                                    arrays[f"{role}/base/{key}"] = value
                                    arrays[f"{role}/proposal/{key}"] = proposals[role][key]
                            np.savez_compressed(output / "aggregation" / f"{variant}-{seed}-{arm}-{round_id}.npz", **arrays)
                            arms[arm] = incumbent.repair(proposals)
                            item["accounting"]["repair_optimization"].append({"seed": seed, "round": round_id,
                                "arm": arm, "roles": repair_accounting(incumbent, arms[arm])})
                            bases[arm] = copy.deepcopy(proposals)
                            gate_before = collect(incumbent, .95, gate_seed, f"r{round_id}/{arm}/incumbent-gate")
                            gate_after = collect(arms[arm], .95, gate_seed, f"r{round_id}/{arm}/candidate-gate")
                            gate = empirical_gate([gate_before], [gate_after])
                            # The existing gate checks events and finite cost; require separately
                            # reconstructed physical validity too before reporting acceptance.
                            if not sound(summary([gate_before])) or not sound(summary([gate_after])):
                                gate.update(accept=False, reason="invalid_or_incomplete_gate_physics")
                            control = summary([collect(incumbent, 1.05, report_seed, f"r{round_id}/{arm}/incumbent-report")])
                            raw = summary([collect(arms[arm], 1.05, report_seed, f"r{round_id}/{arm}/raw-report")])
                            arm_row = {"raw": raw, "deployed": raw if gate["accept"] else control,
                                       "gate": gate, "incumbent": control}
                        else:
                            raw = summary([collect(arms[arm], 1.05, report_seed, f"r{round_id}/{arm}/report")])
                            arm_row = {"raw": raw, "deployed": raw, "gate": None}
                        arm_row["representation_hashes"] = {role: model_hash(values)
                            for role, values in arms[arm].representations().items()}
                        arm_row["learner_updates"] = {"heating": arms[arm].heating.steps, "beam": arms[arm].beam.steps,
                            "heating_actor": delayed_actor_update_count(0, arms[arm].heating.steps,
                                arms[arm].heating.cfg.policy_delay, arms[arm].heating.cfg.actor_warmup_updates),
                            "beam_actor": arms[arm].beam.steps, "beam_entropy_temperature": arms[arm].beam.steps}
                        arm_row["actor_hashes"] = {role: hashlib.sha256(b"".join(
                            tensor.detach().cpu().numpy().tobytes() for tensor in getattr(arms[arm], role).actor.state_dict().values()
                        )).hexdigest() for role in ("heating", "beam")}
                        round_row["arms"][arm] = arm_row
                        seed_row["arms"][arm] = arm_row
                        save()
                    if donor_hashes != {source: pair_state_digest(pair) for source, pair in donors.items()}:
                        raise RuntimeError("Disposable contribution mutated a donor learner")
                if any(arms[arm].heating.steps == 0 or arms[arm].beam.steps == 0 for arm in ARMS):
                    raise RuntimeError("Both TD3 and SAC must actually update in every arm")
                seed_row["status"] = "complete"
                current_seed = None
                save()
            item["status"] = "complete"
        result["status"] = "complete"
        phase = "complete"
        save()
        return result
    except Exception as error:
        failure = {"phase": phase, "type": type(error).__name__, "message": str(error)[:2000]}
        result["failures"].append(failure)
        if current_seed is not None:
            current_seed.update(status="failed", training_stable=False, failure=failure)
        for name in ("baseline", "candidate"):
            if result[name]["status"] == "running":
                result[name]["status"] = "failed"
        result["status"] = "partial" if any(result[name]["seeds"] for name in ("baseline", "candidate")) else "failed"
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if not 1 <= args.max_seconds <= 1800:
        parser.error("--max-seconds must be between 1 and 1800")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)

    def hard_timeout():
        # Separate durable marker avoids racing the main result writer. This
        # watchdog survives a dashboard restart and kills wedged native calls.
        try:
            _write(args.output / "timeout.json", {"status": "failed", "type": "TimeoutError",
                "message": "Worker hard wall-time limit reached", "max_seconds": args.max_seconds})
        finally:
            os._exit(124)

    watchdog = threading.Timer(args.max_seconds, hard_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        run(config, args.reference, args.output, max_seconds=args.max_seconds)
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
