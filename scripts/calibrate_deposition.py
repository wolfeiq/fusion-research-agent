"""Bounded SPARC EC deposition sweep with fixed-command, paired-dt evidence.

Run from the repository root:
  python -m scripts.calibrate_deposition --output results/deposition_live_smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from hfmarl.devices.registry import get
from hfmarl.envs.deposition_env import (
    OBSERVATION_SCHEMA,
    PROFILE_RHO,
    CoupledDepositionEnv,
    DepositionReference,
    sample_temperature_profiles,
    sparc_deposition_task,
)


def run(output: Path, *, episode_length=15.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = sparc_deposition_task(episode_length=episode_length)
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        *(root / "hfmarl" / "envs").glob("*.py"),
        root / "hfmarl" / "devices" / "registry.py",
    ]
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(source_paths):
            archive.write(source, source.relative_to(root).as_posix())
    manifest = {
        "study": "SPARC Gaussian EC deposition development calibration",
        "device": "sparc_like",
        "fixed_dt": [0.1, 0.05],
        "radius": [0.2, 0.45, 0.7],
        "auxiliary_fraction": 0.5,
        "ec_budget_fraction": 0.5,
        "ec_used_fraction": 1.0,
        "episode_length": episode_length,
        "action_window": task.delta_t_a,
        "training_episodes": 0,
        "optimizer_updates": 0,
        "planned_cases": 6,
        "planned_action_windows": 6 * task.steps_per_shot,
        "planned_simulated_seconds": 6 * episode_length,
        "observation_schema": OBSERVATION_SCHEMA,
        "profile_sampling_rho": PROFILE_RHO.tolist(),
        "timestep_screen": "per-sample beta_N max(.001,1%); Te bins max(.005 keV,1%); completion match",
        "reference_choice": "rho=.45, fixed_dt=.05 endpoint, chosen before running",
        "scope": "simulated prescribed Gaussian heating; no mirrors or island dynamics",
        "source_archive_sha256": hashlib.sha256(
            (output / "source.zip").read_bytes()
        ).hexdigest(),
    }
    _save(output / "manifest.json", manifest)
    cases = []
    start = time.perf_counter()
    for fixed_dt in manifest["fixed_dt"]:
        for rho in manifest["radius"]:
            case = {
                "fixed_dt": fixed_dt,
                "rho": rho,
                "status": "initializing",
                "trajectory": [],
            }
            cases.append(case)
            action = np.array([0.0, 0.0, 2 * (rho - 0.05) / 0.8 - 1, 1.0])
            env = CoupledDepositionEnv(
                get("sparc_like"),
                task=task,
                calibration_mode=True,
                config_overrides={"fixed_dt": fixed_dt},
            )
            case["config_sha256"] = hashlib.sha256(
                json.dumps(
                    env.core.config_dict, sort_keys=True, allow_nan=False
                ).encode()
            ).hexdigest()
            case["joint_action"] = action.tolist()
            _save(output / "cases.json", cases)
            try:
                _, case["initial_info"] = env.reset()
                case["initial"] = _point(env)
                for _ in range(task.steps_per_shot):
                    _, reward, terminated, truncated, info = env.step(action)
                    row = _point(env)
                    row.update(
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        violations=list(info.get("violations", ())),
                        allocator=info["allocator"],
                    )
                    case["trajectory"].append(row)
                    if terminated or truncated:
                        case["status"] = "failed" if terminated else "complete"
                        break
                if case["status"] == "initializing":
                    case["status"] = "incomplete"
                if case["status"] == "complete" and (
                    len(case["trajectory"]) != task.steps_per_shot
                    or not np.isclose(env.core.t - env.core.t_initial, episode_length)
                ):
                    case["status"] = "incomplete"
                if case["status"] == "complete" and fixed_dt == 0.05 and rho == 0.45:
                    _, te = sample_temperature_profiles(env.core.read_profiles())
                    reference = DepositionReference(
                        tuple(te),
                        float(env.core.read_scalars()["beta_N"]),
                        tuple(PROFILE_RHO),
                        tuple(action),
                        case["config_sha256"],
                        len(env.trajectory),
                        float(env.core.t - env.core.t_initial),
                        "sparc_like",
                        (),
                    )
                    _save(output / "reference.json", reference.to_dict())
            except Exception as exc:  # noqa: BLE001 -- preserve every attempted scenario and its failure
                case["status"] = "failed"
                case["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                case["commands"] = env.command_history
                _save(output / "cases.json", cases)
    pairs = []
    for rho in manifest["radius"]:
        coarse = next(c for c in cases if c["rho"] == rho and c["fixed_dt"] == 0.1)
        fine = next(c for c in cases if c["rho"] == rho and c["fixed_dt"] == 0.05)
        beta_diffs, te_diffs, flagged = [], [], []
        if coarse["status"] != "complete" or fine["status"] != "complete":
            flagged.append("incomplete_case")
        if len(coarse["trajectory"]) != len(fine["trajectory"]):
            flagged.append("length_mismatch")
        paired_trajectory = (
            zip(coarse["trajectory"], fine["trajectory"]) if not flagged else ()
        )
        for index, (a, b) in enumerate(paired_trajectory):
            beta_diff = abs(a["scalars"]["beta_N"] - b["scalars"]["beta_N"])
            te_diff = np.abs(np.array(a["Te_sampled_keV"]) - b["Te_sampled_keV"])
            beta_diffs.append(beta_diff)
            te_diffs.append(float(te_diff.max()))
            if (
                beta_diff > max(0.001, abs(b["scalars"]["beta_N"]) * 0.01)
                or np.any(
                    te_diff > np.maximum(0.005, np.abs(b["Te_sampled_keV"]) * 0.01)
                )
                or a["time"] != b["time"]
                or a["violations"] != b["violations"]
            ):
                flagged.append(index)
        pairs.append(
            {
                "rho": rho,
                "flagged_samples": flagged,
                "max_beta_difference": max(beta_diffs, default=None),
                "max_Te_bin_difference_keV": max(te_diffs, default=None),
            }
        )
    endpoints = {
        str(c["rho"]): c["trajectory"][-1]["Te_sampled_keV"]
        for c in cases
        if c["fixed_dt"] == 0.05 and c["status"] == "complete"
    }
    separation = None
    if "0.2" in endpoints and "0.7" in endpoints:
        separation = float(
            np.max(np.abs(np.array(endpoints["0.2"]) - endpoints["0.7"]))
        )
    summary = {
        "completed_cases": sum(c["status"] == "complete" for c in cases),
        "attempted_cases": len(cases),
        "attempted_windows": sum(len(c.get("commands", [])) for c in cases),
        "completed_windows": sum(
            sum(p["completion"] == "advanced" for p in c.get("commands", []))
            for c in cases
        ),
        "wall_seconds": time.perf_counter() - start,
        "training_episodes": 0,
        "optimizer_updates": 0,
        "timestep_pairs": pairs,
        "endpoint_Te_separation_keV": separation,
        "interpretation": "Actuator/calibration evidence only; no RL or federation efficacy tested",
    }
    _save(output / "summary.json", summary)
    return summary


def _point(env):
    profiles = env.core.read_profiles()
    point = {
        "time": float(env.core.t),
        "scalars": env.core.read_scalars(),
        "profiles": {key: value.tolist() for key, value in profiles.items()},
    }
    try:
        _, te = sample_temperature_profiles(profiles)
        point["Te_sampled_keV"] = te.tolist()
    except ValueError as exc:
        point.update(Te_sampled_keV=None, invalid_profile=str(exc))
    return point


def _save(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-length", type=float, default=15.0)
    args = parser.parse_args()
    summary = run(args.output, episode_length=args.episode_length)
    print(json.dumps(summary, indent=2))
    if summary["completed_cases"] != 6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
