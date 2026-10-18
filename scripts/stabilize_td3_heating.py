"""Bounded, local live-TORAX heating diagnosis; raw policies, no deployment gate.

The microwave command is fixed at the calibrated location and full budget use
unless coupled mode is explicitly requested. All configurations and cases are
archived before physics. This is development work, never confirmation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
import zipfile
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from hfmarl.devices.registry import get
from hfmarl.envs.deposition_env import CoupledDepositionEnv, sparc_deposition_task
from hfmarl.experiments.coupled_control import LearnerPair, physical_snapshot
from scripts.analyze_coupled_federation import meets_competence, summarize_cases
from scripts.run_coupled_federation import atomic_json, fixed_episode, learner_configs

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("original", "normalized_actions", "critic_burnin", "slow_actor")


def variant_config(name):
    td3, sac = learner_configs()
    changes = {
        "original": {},
        "normalized_actions": {"critic_action_coordinates": "normalized"},
        "critic_burnin": {"actor_warmup_updates": 300},
        "slow_actor": {"actor_lr": 1e-5},
    }
    return replace(td3, **changes[name]), sac


def actor_hash(learner):
    result = hashlib.sha256()
    for name, value in learner.actor.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().numpy().tobytes())
    return result.hexdigest()


def pair_state_digest(pair):
    """Hash state values, never pickle storage identities that can change on read."""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            visit(value.detach().cpu().numpy())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str((type(value).__name__, len(value))).encode())
            for element in value:
                visit(element)
        else:
            digest.update(repr(value).encode())
            digest.update(b"\0")

    for learner in (pair.heating, pair.beam):
        visit(asdict(learner.cfg))
        for name in ("actor", "critics", "target_actor", "target_critics",
                     "actor_opt", "critic_opt", "alpha_opt"):
            if hasattr(learner, name):
                obj = getattr(learner, name)
                visit(obj.state_dict())
                if isinstance(obj, torch.nn.Module):
                    visit([(p.requires_grad, p.grad) for p in obj.parameters()])
        for name in ("rng", "replay_rng"):
            visit(getattr(learner, name).bit_generator.state)
        for name in ("torch_rng", "action_rng"):
            if hasattr(learner, name):
                visit(getattr(learner, name).get_state())
        for name in ("replay", "cursor", "steps", "env_steps", "log_alpha"):
            if hasattr(learner, name):
                visit(getattr(learner, name))
        if hasattr(learner, "log_alpha"):
            visit(learner.log_alpha.grad)
    return digest.hexdigest()


def run_episode(env, pair, seed, *, train=False, warmup=False, coupled=False,
                active="heating"):
    env.rng = np.random.default_rng(seed)
    obs, initial_info = env.reset()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 719]))
    trace, diagnostics, events = [], {"heating": [], "beam": []}, set()
    initial = physical_snapshot(env)

    def heat(x):
        return (rng.uniform(-1, 1, 2).astype(np.float32) if warmup else
                pair.heating.action(x, explore=train and active in {"heating", "both"}))

    heating = heat(obs)
    for _ in range(env.task.steps_per_shot):
        beam_obs = np.concatenate((obs, heating)).astype(np.float32)
        beam = ((rng.uniform(-1, 1, 2).astype(np.float32) if warmup else
                 pair.beam.action(beam_obs, explore=train and active in {"beam", "both"}))
                if coupled else np.array([0., 1.], np.float32))
        action = np.concatenate((heating, beam))
        nxt, reward, terminated, truncated, info = env.step(action)
        terminal = bool(terminated or truncated)
        if not np.isfinite(reward) or not np.isfinite(nxt).all():
            raise FloatingPointError("Invalid trajectory cannot enter replay")
        next_heating = np.zeros(2, np.float32) if terminal else heat(nxt)
        if train:
            pair.heating.remember(obs, heating, reward, nxt, terminal)
            if coupled:
                pair.beam.remember(beam_obs, beam, reward,
                    np.concatenate((nxt, next_heating)).astype(np.float32), terminal)
            for role in ("heating", "beam"):
                if (role == "heating" or coupled) and active in {role, "both"}:
                    diagnostic = getattr(pair, role).train_step()
                    if diagnostic is not None:
                        diagnostics[role].append(diagnostic)
        events.update(info.get("violations", ()))
        trace.append({"action": action.tolist(), "reward": float(reward),
            "components": info.get("reward_components", {}), "allocator": info.get("allocator", {}),
            "violations": list(info.get("violations", ())), "error": info.get("error"),
            "physical": physical_snapshot(env)})
        obs, heating = nxt, next_heating
        if terminal:
            return {"seed": int(seed), "cost": -sum(x["reward"] for x in trace),
                "events": sorted(events), "complete": bool(truncated and not terminated),
                "windows": len(trace), "initial_info": initial_info,
                "initial_physical": initial, "trace": trace,
                "optimizer_updates": {**{k: len(v) for k, v in diagnostics.items()}, "joint": 0},
                "training_diagnostics": diagnostics}
    raise RuntimeError("Environment failed to terminate")


def run(output, reference_path, *, seeds=(17,), variants=VARIANTS, episodes=48,
        checkpoints=(0, 12, 24, 36, 48), factors=(.95, 1.05), coupled=False,
        maximum_seconds=1800, stress_updates=0):
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds)
            or not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS)
            or type(episodes) is not int or episodes < 2 or not checkpoints
            or any(type(c) is not int or c < 0 for c in checkpoints)
            or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
            or checkpoints[0] != 0 or checkpoints[-1] != episodes
            or not factors or len(set(factors)) != len(factors)
            or not all(np.isfinite(f) and .8 <= f <= 1.2 for f in factors)
            or not np.isfinite(maximum_seconds) or maximum_seconds <= 0
            or type(stress_updates) is not int or stress_updates < 0):
        raise ValueError("Invalid bounded study definition")
    reference = json.loads(Path(reference_path).read_text())
    if (reference["device"] != "sparc_like" or reference["sampled_violations"]
            or reference["calibration_simulated_seconds"] != 15.
            or reference["joint_action"] != [0., 0., 0., 1.]):
        raise ValueError("Require the original clean calibrated reference")
    configs = {name: variant_config(name) for name in variants}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = [*sorted((ROOT / "hfmarl").rglob("*.py")), Path(__file__),
        ROOT / "scripts/run_coupled_federation.py", ROOT / "scripts/analyze_coupled_federation.py",
        ROOT / "scripts/analyze_deposition_pilot.py"]
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as packed:
        for file in files:
            packed.write(file, file.relative_to(ROOT).as_posix())
    count = len(seeds) * len(variants)
    manifest = {"study": "local TD3 drift diagnosis", "scope": "development only; no federation or gates",
        "seeds": list(seeds), "variants": list(variants), "episodes_per_learner": episodes,
        "checkpoints": list(checkpoints), "evaluation_factors": list(factors),
        "training_factors": [1.], "warmup_episodes": 2,
        "partner": "alternating SAC" if coupled else "fixed rho action0, full EC budget fraction1",
        "configs": {k: {"td3": asdict(v[0]), "sac": asdict(v[1])} for k, v in configs.items()},
        "reference": reference, "maximum_wall_seconds": maximum_seconds,
        "stress_updates": stress_updates,
        "budget": {"training_episodes": count * episodes,
            "evaluation_episodes": count * (len(checkpoints) + bool(stress_updates)) * len(factors),
            "fixed_reference_episodes": len(factors)},
        "criterion": "Raw learner meets original beta/profile competence at both final two checkpoints on every factor and seed; later failures retained. Initial competence alone is not learned improvement.",
        "source_archive_sha256": hashlib.sha256((output / "source.zip").read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "runtime": {k: version(k) for k in ("torch", "torax", "jax", "numpy")}}
    atomic_json(output / "manifest.json", manifest)
    torch.set_num_threads(1)
    started = time.monotonic()
    task = sparc_deposition_task(target_beta=reference["beta_N"])
    nominal = CoupledDepositionEnv(get("sparc_like"), task=task,
        target_Te_keV=reference["target_Te_keV"], config_overrides={"fixed_dt": .05})
    config_hash = hashlib.sha256(json.dumps(nominal.core.config_dict, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if config_hash != reference["config_sha256"]:
        raise ValueError("Calibration physics mismatch")
    temperature = float(nominal.core.config_dict["profile_conditions"]["T_i"][0.0][0.0])
    envs = {1.: nominal}
    for factor in factors:
        if factor not in envs:
            envs[factor] = CoupledDepositionEnv(get("sparc_like"), task=task,
                target_Te_keV=reference["target_Te_keV"],
                config_overrides={"fixed_dt": .05, "initial_T_keV": temperature * factor})
    bounds = [nominal.allocator.aux.hi / 1e8, nominal.allocator.ec.hi / 1e8]
    ledger, summaries = [], {}

    def execute(label, env, action):
        if time.monotonic() - started > maximum_seconds:
            raise TimeoutError("Declared wall-time budget exhausted")
        path = output / (label + ".json")
        atomic_json(path, {"status": "attempted", "label": label})
        try:
            result = action()
            atomic_json(path, {"status": "completed", "result": result})
            ledger.append({"label": label, "windows": result["windows"],
                "optimizer_updates": result["optimizer_updates"]})
            atomic_json(output / "work.json", ledger)
            return result
        except Exception as error:
            atomic_json(path, {"status": "failed", "error": repr(error)})
            raise

    def evaluate(pair, label, seed):
        before = pair_state_digest(pair)
        cases = []
        for i, factor in enumerate(factors):
            result = execute(f"{label}-case{i}", envs[factor], lambda factor=factor, i=i: run_episode(
                envs[factor], pair, seed + 400000 + i, coupled=coupled))
            result["scenario"] = {"initial_temperature_factor": factor}
            cases.append(result)
        if pair_state_digest(pair) != before:
            raise RuntimeError("Evaluation mutated learner state")
        summary = summarize_cases(cases, reference)
        summary["competent"] = meets_competence(summary)
        summary["actor_sha256"] = actor_hash(pair.heating)
        summary["updates"] = pair.heating.steps
        summary["mean_aux_MW"] = float(np.mean([x["allocator"]["aux_power_W"] / 1e6 for c in cases for x in c["trace"]]))
        summary["mean_ec_MW"] = float(np.mean([x["allocator"]["ec_power_W"] / 1e6 for c in cases for x in c["trace"]]))
        print(json.dumps({"case": label, "cost": summary["mean_cost"],
            "profile_rmse": summary["mean_Te_profile_rmse_keV"], "competent": summary["competent"],
            "aux_MW": summary["mean_aux_MW"]}), flush=True)
        return summary

    for i, factor in enumerate(factors):
        execute(f"reference-case{i}", envs[factor], lambda factor=factor, i=i: fixed_episode(
            envs[factor], reference["joint_action"], 400000 + i))
    try:
        for seed in seeds:
            for variant in variants:
                td3, sac = configs[variant]
                pair = LearnerPair(seed, td3, sac, bounds)
                key = f"{variant}-seed{seed}"
                summaries[key] = {"checkpoints": {}}
                for count in range(episodes + 1):
                    if count in checkpoints:
                        summaries[key]["checkpoints"][str(count)] = evaluate(pair, f"{key}-eval{count}", seed)
                        atomic_json(output / "summary.json", {"status": "running", "results": summaries})
                    if count == episodes:
                        break
                    active = "heating" if not coupled or count % 2 == 0 else "beam"
                    execute(f"{key}-train{count}", nominal, lambda pair=pair, seed=seed, count=count, active=active: run_episode(
                        nominal, pair, seed + count, train=True, warmup=count < 2,
                        coupled=coupled, active=active))
                checkpoint = output / f"{key}-trusted.pkl"
                checkpoint.write_bytes(pickle.dumps(pair, protocol=5))
                summaries[key]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                if stress_updates:
                    diagnostics = [pair.heating.train_step() for _ in range(stress_updates)]
                    atomic_json(output / f"{key}-stress-updates.json", diagnostics)
                    summaries[key]["stress"] = evaluate(pair, f"{key}-stress", seed)
        result = {"status": "completed", "results": summaries, "elapsed_seconds": time.monotonic() - started,
            "actual_episodes": len(ledger), "actual_windows": sum(x["windows"] for x in ledger)}
        atomic_json(output / "summary.json", result)
        return result
    except Exception as error:
        atomic_json(output / "failure.json", {"error": repr(error), "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/deposition_calibration_verified/reference.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 12, 24, 36, 48])
    parser.add_argument("--factors", nargs="+", type=float, default=[.95, 1.05])
    parser.add_argument("--coupled", action="store_true")
    parser.add_argument("--maximum-seconds", type=int, default=1800)
    parser.add_argument("--stress-updates", type=int, default=0)
    args = parser.parse_args()
    run(args.output, args.reference, seeds=tuple(args.seeds), variants=tuple(args.variants),
        episodes=args.episodes, checkpoints=tuple(args.checkpoints), factors=tuple(args.factors),
        coupled=args.coupled, maximum_seconds=args.maximum_seconds, stress_updates=args.stress_updates)


if __name__ == "__main__":
    main()
