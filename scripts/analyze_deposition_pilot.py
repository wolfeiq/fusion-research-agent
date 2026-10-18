"""Recompute the small deposition pilot's costs and physical errors from traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def score(result, reference):
    trace = result["trace"]
    cost = -sum(float(row["reward"]) for row in trace)
    if not np.isclose(cost, result["cost"], rtol=0, atol=1e-10):
        raise ValueError("Saved cost disagrees with trajectory rewards")
    if len(trace) != result["windows"]:
        raise ValueError("Saved window count disagrees with trajectory")
    beta_errors, profile_squared_errors = [], []
    for row in trace:
        physical = row.get("physical", row)
        if physical.get("read_errors") or physical.get("nonfinite_fields"):
            raise ValueError("Invalid physical state requires a separate failure report")
        beta = float(physical["scalars"]["beta_N"])
        profiles = physical["profiles"]
        rho, values = np.asarray(profiles["rho_norm"]), np.asarray(profiles["T_e"])
        if (not np.isfinite(beta) or rho.ndim != 1 or rho.size < 2
                or values.shape != rho.shape or not np.isfinite(rho).all()
                or not np.isfinite(values).all() or np.any(np.diff(rho) <= 0)
                or rho[0] > min(reference["profile_rho"])
                or rho[-1] < max(reference["profile_rho"])):
            raise ValueError("Invalid physical profile or pressure")
        beta_errors.append(abs(beta - reference["beta_N"]))
        te = np.interp(reference["profile_rho"], rho, values)
        profile_squared_errors.extend(np.square(te - reference["target_Te_keV"]).tolist())
    return {"cost": cost, "beta_mae": float(np.mean(beta_errors)),
            "Te_profile_rmse_keV": float(np.sqrt(np.mean(profile_squared_errors))),
            "complete": result["complete"], "events": result["events"]}


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    training = json.loads((directory / "training.json").read_text())
    evaluation = json.loads((directory / "evaluation.json").read_text())
    initial_path = directory / "initial_evaluation.json"
    has_initial_phase = "initial_policy_evaluation" in manifest
    if initial_path.exists() != has_initial_phase:
        raise ValueError("Initial evaluation artifact and frozen manifest disagree")
    initial = json.loads(initial_path.read_text()) if has_initial_phase else {}
    archive_hash = hashlib.sha256((directory / "source.zip").read_bytes()).hexdigest()
    if archive_hash != manifest["source_archive_sha256"] or summary["status"] != "complete":
        raise ValueError("Missing completion or source archive mismatch")
    if len(training) != 3 * manifest["training_episodes_per_arm"]:
        raise ValueError("Incomplete training matrix")
    if set(evaluation) != set(manifest["arms"]) | {manifest["comparator"]}:
        raise ValueError("Incomplete controller comparison")
    results = {}
    for arm, rows in evaluation.items():
        if [row["factor"] for row in rows] != manifest["evaluation_factors"]:
            raise ValueError("Changed evaluation conditions")
        results[arm] = [{"initial_temperature_factor": row["factor"],
                         **score(row["result"], manifest["reference"])} for row in rows]
        if any(row["status"] != "complete" for row in rows):
            raise ValueError("Attempted evaluation cannot be scored as complete")
        np.testing.assert_allclose([r["cost"] for r in results[arm]],
                                   summary["evaluation"][arm]["costs"], rtol=0, atol=1e-10)
    initial_scores = {}
    if has_initial_phase:
        if set(initial) != set(manifest["arms"]):
            raise ValueError("Incomplete initial-policy comparison")
        for arm, rows in initial.items():
            if [row["factor"] for row in rows] != manifest["evaluation_factors"]:
                raise ValueError("Changed initial evaluation conditions")
            if any(row["status"] != "complete" for row in rows):
                raise ValueError("Attempted initial evaluation cannot be scored as complete")
            if any(any(row["result"].get("optimizer_updates", {}).values()) for row in rows):
                raise ValueError("Initial policy evaluation must not update the learners")
            initial_scores[arm] = [{"initial_temperature_factor": row["factor"],
                                    **score(row["result"], manifest["reference"])} for row in rows]
            np.testing.assert_allclose([r["cost"] for r in initial_scores[arm]],
                                       summary["initial_evaluation"][arm]["costs"], rtol=0, atol=1e-10)
    training_scores = {arm: [] for arm in manifest["arms"]}
    for row in training:
        if row["status"] != "complete":
            raise ValueError("Training attempt did not complete")
        training_scores[row["arm"]].append(score(row["result"], manifest["reference"]))
    phase_cases = {"training": len(training),
                   "evaluation": sum(map(len, evaluation.values())),
                   "initial_evaluation": sum(map(len, initial.values()))}
    phase_windows = {
        "training": sum(row["result"]["windows"] for row in training),
        "evaluation": sum(row["result"]["windows"] for rows in evaluation.values() for row in rows),
        "initial_evaluation": sum(row["result"]["windows"] for rows in initial.values() for row in rows),
    }
    for phase, cases in phase_cases.items():
        if (phase_windows[phase] != summary.get(phase + "_windows", 0)
                or cases != summary.get(phase + "_episodes", 0)):
            raise ValueError("Summary work accounting differs from saved cases")
    total_windows = sum(phase_windows.values())
    if (total_windows > manifest.get("maximum_windows", total_windows)
            or sum(phase_cases.values()) > manifest.get("maximum_cases", sum(phase_cases.values()))):
        raise ValueError("Saved work exceeds frozen experiment budget")
    baseline = results[manifest["comparator"]]
    comparisons = {
        arm: {"lower_cost_than_fixed": sum(a["cost"] < b["cost"] for a, b in zip(rows, baseline)),
              "lower_beta_mae_than_fixed": sum(a["beta_mae"] < b["beta_mae"] for a, b in zip(rows, baseline)),
              "lower_profile_rmse_than_fixed": sum(a["Te_profile_rmse_keV"] < b["Te_profile_rmse_keV"]
                                                    for a, b in zip(rows, baseline))}
        for arm, rows in results.items() if arm != manifest["comparator"]
    }
    before_after = {
        arm: [{"initial_temperature_factor": before["initial_temperature_factor"],
               "initial": {metric: before[metric] for metric in ("cost", "beta_mae", "Te_profile_rmse_keV")},
               "trained": {metric: after[metric] for metric in ("cost", "beta_mae", "Te_profile_rmse_keV")},
               "trained_minus_initial": {metric: after[metric] - before[metric]
                                         for metric in ("cost", "beta_mae", "Te_profile_rmse_keV")},
               "initial_complete": before["complete"], "trained_complete": after["complete"],
               "initial_events": before["events"], "trained_events": after["events"]}
              for before, after in zip(rows, results[arm])]
        for arm, rows in initial_scores.items()
    }
    report = {"scope": "one training seed; three deterministic initial conditions, not independent replicates",
              "source_archive_verified": True, "evaluation": results,
              "comparisons": comparisons, "total_action_windows": total_windows,
              "total_cases": sum(phase_cases.values()), "cases_by_phase": phase_cases,
              "action_windows_by_phase": phase_windows,
              "initial_evaluation": initial_scores, "before_after_training": before_after,
              "frozen_comparison": "Same saved factors, pressure/profile target, deterministic action rule and reward before/after training; no selection of favorable cases",
              "initialization": manifest.get("initialization_change", "legacy pilot; no initial-policy evaluation"),
              "training_optimizer_updates": summary["training_optimizer_updates"],
              "wall_seconds": summary["wall_seconds"],
              "federation_rounds": 0, "confirmation_data_used": False}
    (directory / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plot(directory, training_scores, results, initial_scores)
    return report


def plot(directory, training, evaluation, initial=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"sac_fixed_heating": "SAC / fixed heating", "coupled_alternating": "TD3 + SAC",
              "joint_sac": "Single joint SAC", "calibration_fixed_command": "Fixed calibrated"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for arm, rows in training.items():
        axes[0].plot(np.arange(1, len(rows) + 1), [r["cost"] for r in rows], marker="o", label=labels[arm])
    axes[0].set(title="Training cost (exploration included)", xlabel="Training episode", ylabel="Integrated cost; lower is better")
    axes[0].legend(fontsize=8)
    for axis, metric, title in [(axes[1], "cost", "Deterministic evaluation cost"),
                                (axes[2], "Te_profile_rmse_keV", "Temperature-profile RMSE (keV)")]:
        for index, (arm, rows) in enumerate(evaluation.items()):
            axis.scatter(np.full(len(rows), index), [r[metric] for r in rows], s=32)
            if initial and arm in initial:
                axis.scatter(np.full(len(initial[arm]), index),
                             [r[metric] for r in initial[arm]], s=48,
                             facecolors="none", edgecolors="black", marker="s",
                             label="Initial policy" if index == 0 else None)
        if initial:
            axis.legend(fontsize=8)
        axis.set_xticks(range(len(evaluation)), [labels[arm] for arm in evaluation], rotation=25, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("Lower is better")
    fig.suptitle("Short TORAX development pilot: one seed, three initial conditions — no federation", fontsize=12)
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
