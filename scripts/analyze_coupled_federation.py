"""Audit repeated coupled-RL development evidence without loading local policies.

Only JSON, NumPy coordinator tensors, and source archives are read. The analysis
keeps raw federation, reset controls and the gated deployment branch separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from itertools import pairwise
from pathlib import Path

import numpy as np

from hfmarl.envs.limits import DEFAULT_LIMITS, LimitSet
from hfmarl.federation.representation import model_hash
from scripts.analyze_deposition_pilot import score

ROLES = ("heating_td3", "deposition_sac")
TRAIN_ARMS = ("local", "reset_only", "initial_base_pull", "raw_federated", "deployed", "joint_sac")
EVAL_ARMS = (*TRAIN_ARMS, "calibration_fixed_command")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def episode_metrics(result, reference, beta_tolerance=.005):
    """Preserve failed cases; corrupt accounting is an error, never a result."""
    trace = result["trace"]
    if not trace:
        raise ValueError("An episode without observed windows is not a completed result")
    cost = -sum(float(row["reward"]) for row in trace)
    if (not np.isfinite(cost) or not np.isclose(cost, result["cost"], rtol=0, atol=1e-10)
            or len(trace) != result["windows"]):
        raise ValueError("Episode cost or work accounting disagrees with saved trajectory")
    invalid = [index for index, row in enumerate(trace)
               if row.get("error") or (row.get("physical") or row).get("read_errors")
               or (row.get("physical") or row).get("nonfinite_fields")]
    observed_events = set()
    limits = LimitSet(DEFAULT_LIMITS)
    for index, row in enumerate(trace):
        observed_events.update(row.get("violations", []))
        values = (row.get("physical") or row).get("scalars", {})
        if any(name not in values or values[name] is None or not np.isfinite(values[name])
               for name in ("beta_N", "q95", "fgw_n_e_line_avg")):
            invalid.append(index)
            continue
        checked = limits.evaluate({"beta_N": values["beta_N"], "q95": values["q95"],
                                   "greenwald_fraction": values["fgw_n_e_line_avg"]})
        observed_events.update(checked.violations)
    if not observed_events.issubset(set(result["events"])):
        raise ValueError("Saved event summary hides a trajectory limit event")
    invalid = sorted(set(invalid))
    physical = score(result, reference) if not invalid else {}
    within = None if invalid else float(np.mean([
        abs(float((row.get("physical") or row)["scalars"]["beta_N"]) - reference["beta_N"])
        <= beta_tolerance for row in trace]))
    return {"cost": cost, "complete": bool(result["complete"]),
            "events": list(result["events"]), "windows": len(trace),
            "invalid_physical_windows": invalid,
            "beta_mae": physical.get("beta_mae"),
            "Te_profile_rmse_keV": physical.get("Te_profile_rmse_keV"),
            "beta_within_tolerance_fraction": within,
            "seed": result.get("seed"),
            "scenario": result.get("scenario"),
            "optimizer_updates": result.get("optimizer_updates", {})}


def summarize_cases(cases, reference, beta_tolerance=.005):
    if not cases:
        raise ValueError("An expected case matrix is empty")
    rows = [episode_metrics(case, reference, beta_tolerance) for case in cases]
    result = {"cases": len(rows), "complete_cases": sum(row["complete"] for row in rows),
              "cases_with_events": sum(bool(row["events"]) for row in rows),
              "cases_with_invalid_physics": sum(bool(row["invalid_physical_windows"]) for row in rows),
              "windows": sum(row["windows"] for row in rows), "per_case": rows}
    for metric in ("cost", "beta_mae", "Te_profile_rmse_keV"):
        values = [row[metric] for row in rows]
        # Do not silently discard failed physical readings from a mean.
        result["mean_" + metric] = (float(np.mean(values))
                                    if all(value is not None for value in values) else None)
    result["optimizer_updates"] = {
        role: sum(row["optimizer_updates"].get(role, 0) for row in rows)
        for role in ("heating", "beam", "joint")}
    return result


def meets_competence(summary, *, windows=30, beta_mae=.005,
                     beta_coverage=.9, profile_rmse_keV=.02):
    """Frozen development criterion on every reporting condition, not a mean."""
    return bool(summary["cases"] and all(
        row["complete"] and row["windows"] == windows and not row["events"]
        and not row["invalid_physical_windows"]
        and row["beta_mae"] is not None and row["beta_mae"] <= beta_mae
        and row["beta_within_tolerance_fraction"] is not None
        and row["beta_within_tolerance_fraction"] >= beta_coverage
        and row["Te_profile_rmse_keV"] is not None
        and row["Te_profile_rmse_keV"] <= profile_rmse_keV
        for row in summary["per_case"]))


def competence_intervals(checkpoints):
    """Checkpoint crossings are interval censored; no invented shot precision.

    Two successive observed passes confirm the stated development rule. Later
    loss of competence is reported too; this is not a permanent guarantee.
    """
    if not checkpoints:
        raise ValueError("Competence requires the initial and observed checkpoints")
    shots = [point["target_training_shots"] for point in checkpoints]
    if shots[0] != 0 or any(a >= b for a, b in pairwise(shots)):
        raise ValueError("Checkpoint training shots must start at zero and strictly increase")
    passed = [bool(point["competent"]) for point in checkpoints]
    first = next((index for index, value in enumerate(passed) if value), None)
    confirmed = next((index for index in range(1, len(passed))
                      if passed[index - 1] and passed[index]), None)

    def crossing(index):
        if index is None:
            return {"observed": False, "right_censored_after_training_shots": shots[-1]}
        return {"observed": True, "first_observed_training_shots": shots[index],
                "lower_training_shots": shots[index - 1] if index else 0,
                "upper_training_shots": shots[index], "lower_open": bool(index),
                "already_competent_at_initial_evaluation": index == 0}

    sustained = crossing(None if confirmed is None else confirmed - 1)
    sustained["confirmation_training_shots"] = None if confirmed is None else shots[confirmed]
    sustained["lost_competence_after_confirmation"] = bool(
        confirmed is not None and any(not value for value in passed[confirmed + 1:]))
    return {"first_observed": crossing(first), "two_consecutive_checkpoints": sustained,
            "checkpoint_count": len(checkpoints),
            "caveat": "Only evaluated checkpoints are observed; training, repair, gate and evaluation costs are distinct"}


def paired_comparison(treatment, control):
    if (treatment["cases"] != control["cases"]
            or [row["scenario"] for row in treatment["per_case"]]
            != [row["scenario"] for row in control["per_case"]]
            or [row["seed"] for row in treatment["per_case"]]
            != [row["seed"] for row in control["per_case"]]):
        raise ValueError("A paired comparison requires identical ordered physical scenarios")
    result = {"paired_conditions": treatment["cases"],
              "treatment_complete_cases": treatment["complete_cases"],
              "control_complete_cases": control["complete_cases"],
              "treatment_event_cases": treatment["cases_with_events"],
              "control_event_cases": control["cases_with_events"],
              "treatment_cost": treatment["mean_cost"], "control_cost": control["mean_cost"],
              "treatment_minus_control_cost": treatment["mean_cost"] - control["mean_cost"]}
    for metric in ("beta_mae", "Te_profile_rmse_keV"):
        a, b = treatment["mean_" + metric], control["mean_" + metric]
        result["treatment_minus_control_" + metric] = None if a is None or b is None else a - b
    differences = [a["cost"] - b["cost"] for a, b in zip(treatment["per_case"], control["per_case"])]
    result.update(lower_cost_conditions=sum(x < -1e-12 for x in differences),
                  tied_cost_conditions=sum(abs(x) <= 1e-12 for x in differences),
                  higher_cost_conditions=sum(x > 1e-12 for x in differences))
    return result


def audit_coordinator(path, *, cohort, rounds, strategy):
    """Verify final tensors, role separation, consumed update IDs and full acks."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if (meta["format"] != 1 or meta["version"] != rounds or meta["pending"]
                or meta["strategy"] != strategy or set(meta["channels"]) != set(ROLES)):
            raise ValueError("Coordinator is incomplete or does not match the frozen job")
        if len(meta["history"]) != rounds:
            raise ValueError("Coordinator round history is incomplete")
        schemas = []
        for role in ROLES:
            channel = meta["channels"][role]
            if (set(channel["cohort"]) != set(cohort)
                    or len(channel["cohort"]) != len(cohort) or channel["version"] != rounds):
                raise ValueError("Coordinator channel cohort or version mismatch")
            values = {key: archive[f"{role}/{key}"].copy() for key in channel["keys"]}
            if (not values or any(not np.isfinite(value).all() for value in values.values())
                    or model_hash(values) != channel["hash"]):
                raise ValueError("Coordinator tensor hash or numeric integrity mismatch")
            if {key.split(".", 1)[0] for key in values} != {"q0", "q1"}:
                raise ValueError("Both separate critic trunks are required")
            expected_ids = {f"{meta['run_id']}:{round_id}:{source}:{role}"
                            for round_id in range(1, rounds + 1) for source in cohort}
            if (set(channel["consumed"]) != expected_ids
                    or len(channel["consumed"]) != len(expected_ids)):
                raise ValueError("Missing or duplicate consumed client updates")
            schemas.append(channel["schema"])
        if len(set(schemas)) != len(ROLES):
            raise ValueError("Heating and deposition channels must have different schemas")
    for round_id, row in enumerate(meta["history"], 1):
        if (row["round"] != round_id or row["strategy"] != strategy
                or set(row["acknowledgements"] or {}) != set(cohort)
                or set(row["channels"]) != set(ROLES)):
            raise ValueError("Missing, misordered or incomplete round acknowledgement")
        if any(report["round"] != round_id for report in row["acknowledgements"].values()):
            raise ValueError("Acknowledged client report names the wrong round")
    return meta


def audit_sources(directory, manifest):
    identity_payload = {key: value for key, value in manifest.items() if key != "identity"}
    if digest(identity_payload) != manifest["identity"]:
        raise ValueError("Frozen manifest identity mismatch")
    path = Path(directory) / "source.zip"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["source_archive_sha256"]:
        raise ValueError("Source archive hash mismatch")
    with zipfile.ZipFile(path) as archive:
        expected = set(manifest["source_file_sha256"]) | {"calibration-reference.json"}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("Source archive file set mismatch")
        for name, checksum in manifest["source_file_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"Archived source checksum mismatch: {name}")
        reference_bytes = archive.read("calibration-reference.json")
        if (hashlib.sha256(reference_bytes).hexdigest() != manifest["reference_sha256"]
                or json.loads(reference_bytes) != manifest["reference"]):
            raise ValueError("Calibration reference provenance mismatch")
    return {"manifest_identity": manifest["identity"], "archive_sha256": manifest["source_archive_sha256"],
            "source_files": len(manifest["source_file_sha256"]), "verified": True}


def audit_job_contract(meta, job, manifest):
    config = manifest["config"]
    cohort = [row[0] for row in config["facility_factors"]]
    if (job["seed"] not in config["seeds"] or job["strategy"] not in config["strategies"]
            or job["cohort"] != cohort or job["rounds"] != config["rounds"]
            or job["facility_factors"] != dict(config["facility_factors"])
            or job["revision"] != manifest["identity"]
            or job["run_id"] != f"{manifest['identity']}:{job['strategy']}:seed-{job['seed']}"
            or meta["run_id"] != job["run_id"]):
        raise ValueError("Job does not match frozen study identity or cohort")
    if meta["thresholds"] != (job["thresholds"] if job["strategy"] == "clipped" else None):
        raise ValueError("Coordinator clipping thresholds differ from frozen job")
    clipping = []
    for role in ROLES:
        channel = meta["channels"][role]
        if (channel["schema"] != job["schemas"][role]
                or channel["revisions"] != {name: manifest["identity"] for name in cohort}):
            raise ValueError("Role schema or source revision mismatch")
        previous = job["initial_base_hashes"][role]
        for row in meta["history"]:
            diagnostic = row["channels"][role]
            if diagnostic["base_hash"] != previous:
                raise ValueError("Coordinator base/proposal hash chain is broken")
            if job["strategy"] == "clipped":
                if (diagnostic["thresholds"] != job["thresholds"][role]
                        or diagnostic["source_ids"] != cohort):
                    raise ValueError("Clipping provenance changed during study")
                expected_ids = [f"{job['run_id']}:{row['round']}:{source}:{role}" for source in cohort]
                if diagnostic["update_ids"] != expected_ids:
                    raise ValueError("Clipping diagnostics name the wrong source updates")
                for source, factors in diagnostic["clipping_factors"].items():
                    for branch, factor in factors.items():
                        norm = diagnostic["source_norms"][source][branch]
                        bound = job["thresholds"][role][branch]
                        expected_factor = bound / norm if norm > bound else 1.
                        if not np.isclose(factor, expected_factor, rtol=1e-12, atol=0):
                            raise ValueError("Clipping factor does not implement frozen norm bound")
                        clipping.append({"round": row["round"], "role": role, "source": source,
                                         "branch": branch, "factor": factor, "active": factor < 1.})
                previous = diagnostic["proposal_hashes"]["clipped"]
            else:
                previous = diagnostic["proposal_hash"]
        if previous != channel["hash"]:
            raise ValueError("Final coordinator tensors do not match acknowledged proposal")
    return clipping


def gate_accepts(control, candidate, config):
    """Independent implementation of the finite-condition acceptance rule."""
    if len(control) != len(candidate) or not control:
        return False
    for a, b in zip(control, candidate):
        if (a["scenario"] != b["scenario"] or not a["complete"] or not b["complete"]
                or a["windows"] != 30 or b["windows"] != 30 or a["events"] or b["events"]
                or not np.isfinite([a["cost"], b["cost"]]).all()):
            return False
        if ((b["cost"] - a["cost"]) / max(abs(a["cost"]), config["gate_cost_floor"])
                > config["gate_tolerance"]):
            return False
    return True


def _add_work(total, value):
    for key, count in value.items():
        if isinstance(count, dict):
            _add_work(total.setdefault(key, {}), count)
        else:
            if not isinstance(count, int) or count < 0:
                raise ValueError("Optimizer work must be a nonnegative integer count")
            total[key] = total.get(key, 0) + count


def audit_client(directory, source, manifest, coordinator, seed):
    """Read one client's completed evidence and cross-check every attempt."""
    directory = Path(directory)
    config, reference = manifest["config"], manifest["reference"]
    arms = tuple(manifest["arms"])
    evaluation_arms = (*arms, manifest["comparator"])
    factor = dict(config["facility_factors"])[source]
    pretrain_count = 0 if source == config["target_source"] else config["donor_pretrain_episodes"]
    ledger, counters, work = {}, {}, {"contribution": {}, "repair": {}, "online": {}}

    def collect(cases, phase, arm, round_id, factors, start_index=0):
        if len(cases) != len(factors):
            raise ValueError(f"Missing cases for {source}/{phase}/{arm}/{round_id}")
        for offset, (case, expected_factor) in enumerate(zip(cases, factors)):
            scenario_phase = "gate" if phase.startswith("gate") else phase
            if case.get("scenario") != {"initial_temperature_factor": expected_factor, "phase": scenario_phase}:
                raise ValueError("Physical case does not match frozen scenario matrix")
            index = start_index + offset
            expected_seed = (seed + index if phase == "pretrain" else seed + index + pretrain_count if phase == "training"
                             else seed + 200000 + round_id * 1000 + index if phase.startswith("gate")
                             else seed + 400000 + round_id * 1000 + index)
            if case["seed"] != expected_seed:
                raise ValueError("Case seed differs from the frozen paired seed schedule")
            key = f"r{round_id:03d}-{phase}-{arm}-{index:04d}.json"
            if key in ledger:
                raise ValueError("Case occurs twice in completed evidence")
            ledger[key] = {"sha256": digest(case), "seed": case["seed"], "factor": expected_factor,
                           "round": round_id, "phase": phase, "arm": arm, "index": index}
        if not cases:
            return {"cases": 0, "windows": 0, "optimizer_updates": {}}
        summary = summarize_cases(cases, reference, config["beta_mae_tolerance"])
        category = ("donor_pretraining" if phase == "pretrain" else "initial_evaluation" if phase == "initial"
                    else "round_training" if phase == "training" else "gate" if phase.startswith("gate") else "reporting")
        counter = counters.setdefault(category, {"episodes": 0, "windows": 0, "incomplete_episodes": 0,
                                                  "event_episodes": 0, "invalid_physics_episodes": 0, "by_arm": {}})
        counter["episodes"] += summary["cases"]
        counter["windows"] += summary["windows"]
        counter["incomplete_episodes"] += summary["cases"] - summary["complete_cases"]
        counter["event_episodes"] += summary["cases_with_events"]
        counter["invalid_physics_episodes"] += summary["cases_with_invalid_physics"]
        counter["by_arm"][arm] = counter["by_arm"].get(arm, 0) + summary["cases"]
        _add_work(work["online"].setdefault(category, {}), summary["optimizer_updates"])
        if phase not in {"training", "pretrain"} and any(summary["optimizer_updates"].values()):
            raise ValueError("Deterministic evaluation unexpectedly trained a learner")
        return summary

    pretraining = read_json(directory / "pretraining.json")
    if set(pretraining) != {"coupled_pair", "joint_sac"}:
        raise ValueError("Donor pretraining controller matrix mismatch")
    pretraining_summary = {arm: collect(cases, "pretrain", arm, 0, [factor] * pretrain_count)
                           for arm, cases in pretraining.items()}
    initial = read_json(directory / "initial-evaluation.json")
    if (initial["source"] != source or initial["pretraining_shots_per_controller"] != pretrain_count
            or initial["target_training_shots"] != 0 or set(initial["evaluation"]) != set(evaluation_arms)):
        raise ValueError("Initial evaluation does not match source warmstart contract")
    reports = [factor * mult for mult in config["report_factors"]]
    gates = [factor * mult for mult in config["gate_factors"]]
    initial_summary = {arm: collect(cases, "initial", arm, 0, reports)
                       for arm, cases in initial["evaluation"].items()}
    if (initial["episodes"] != sum(row["cases"] for row in initial_summary.values())
            or initial["windows"] != sum(row["windows"] for row in initial_summary.values())):
        raise ValueError("Initial evaluation accounting mismatch")
    rounds = []
    for round_id in range(1, config["rounds"] + 1):
        record = read_json(directory / f"round-{round_id}.json")
        if (record["round"] != round_id or set(record["training"]) != set(arms)
                or set(record["evaluation"]) != set(evaluation_arms)
                or record["report"] != coordinator["history"][round_id - 1]["acknowledgements"][source]):
            raise ValueError("Client round differs from acknowledged coordinator evidence")
        training = {arm: collect(cases, "training", arm, round_id,
                                [factor] * config["episodes_per_round"],
                                (round_id - 1) * config["episodes_per_round"])
                    for arm, cases in record["training"].items()}
        control = collect(record["gate_control"], "gate_control", "control", round_id, gates)
        candidate = collect(record["gate_candidate"], "gate_candidate", "candidate", round_id, gates)
        evaluation = {arm: collect(cases, "report", arm, round_id, reports)
                      for arm, cases in record["evaluation"].items()}
        report = record["report"]
        if report.get("gate_reference", "repaired_control") != config.get("gate_reference", "repaired_control"):
            raise ValueError("Gate reference differs from the frozen protocol")
        if (report["training_episodes"] != sum(row["cases"] for row in training.values())
                or report["gate_episodes"] != control["cases"] + candidate["cases"]
                or report["evaluation_episodes"] != sum(row["cases"] for row in evaluation.values())
                or not np.isclose(report["deployed_cost"], evaluation["deployed"]["mean_cost"], rtol=0, atol=1e-10)
                or report["raw_costs"] != {arm: [case["cost"] for case in cases] for arm, cases in record["evaluation"].items()}
                or report["raw_events"] != {arm: [case["events"] for case in cases] for arm, cases in record["evaluation"].items()}):
            raise ValueError("Acknowledged metrics do not match complete physical records")
        accepted = gate_accepts(record["gate_control"], record["gate_candidate"], config)
        if report["gate"]["accept"] != accepted:
            raise ValueError("Recorded deployment decision violates frozen gate")
        if report["contribution_optimization"] != record["contribution_optimization"]:
            raise ValueError("Contribution optimizer accounting mismatch")
        _add_work(work["contribution"], record["contribution_optimization"])
        _add_work(work["repair"], report["repair_optimization"])
        comparisons = {f"raw_federated_vs_{arm}": paired_comparison(evaluation["raw_federated"], evaluation[arm])
                       for arm in ("local", "reset_only", "initial_base_pull", "joint_sac") if arm in evaluation}
        comparisons["deployed_vs_local"] = paired_comparison(evaluation["deployed"], evaluation["local"])
        rounds.append({"round": round_id, "additional_training_shots_per_arm": round_id * config["episodes_per_round"],
                       "evaluation": evaluation, "paired_comparisons": comparisons,
                       "gate": {"accepted": accepted, "control": control, "candidate": candidate}})
    attempt_paths = sorted((directory / "attempts").glob("*.json"))
    if {path.name for path in attempt_paths} != set(ledger):
        raise ValueError("Attempt ledger contains missing, failed or unreported work")
    for path in attempt_paths:
        attempt, expected = read_json(path), ledger[path.name]
        context = attempt["context"]
        if (attempt["status"] != "complete" or digest(attempt["result"]) != expected["sha256"]
                or context["source"] != source
                or any(context[name] != expected[name] for name in ("round", "phase", "arm", "index", "seed"))
                or context["initial_temperature_factor"] != expected["factor"]
                or context["training"] != (expected["phase"] in {"pretrain", "training"})):
            raise ValueError("Attempt ledger and committed experiment evidence disagree")
    target_curves = {}
    if source == config["target_source"]:
        for arm in evaluation_arms:
            checkpoints = []
            for index, summary in enumerate([initial_summary[arm], *[row["evaluation"][arm] for row in rounds]]):
                competent = meets_competence(summary, beta_mae=config["beta_mae_tolerance"],
                    beta_coverage=config["beta_coverage"], profile_rmse_keV=config["profile_rmse_tolerance_keV"])
                checkpoints.append({"target_training_shots": index * config["episodes_per_round"],
                                    "actual_controller_training_shots": (0 if arm == manifest["comparator"] else index * config["episodes_per_round"]),
                                    "competent": competent, "evaluation": summary})
            target_curves[arm] = {"checkpoints": checkpoints,
                                  "crossings": (None if arm == manifest["comparator"] else competence_intervals(checkpoints))}
    return {"source": source, "source_pretraining_shots_per_controller": pretrain_count,
            "pretraining": pretraining_summary, "initial_evaluation": initial_summary,
            "rounds": rounds, "cold_target_curves": target_curves,
            "work": counters, "optimizer_work": work, "attempts_verified": len(ledger)}


def incomplete_attempts(directory, reference, beta_tolerance):
    """Preserve failed/pending work; do not turn partial jobs into efficacy results."""
    result = {"complete_attempts": 0, "observed_windows": 0, "failed_or_uncertain": []}
    for path in sorted(Path(directory).glob("clients/*/attempts/*.json")):
        row = read_json(path)
        if row["status"] == "complete":
            measured = episode_metrics(row["result"], reference, beta_tolerance)
            result["complete_attempts"] += 1
            result["observed_windows"] += measured["windows"]
        else:
            result["failed_or_uncertain"].append({"file": path.relative_to(directory).as_posix(),
                "status": row["status"], "context": row["context"], "error": row.get("error"),
                "last_physical_state": row.get("last_physical_state")})
    result["caveat"] = "Uncommitted attempts may have consumed additional simulator or optimizer work; observed counts are lower bounds"
    return result


def target_findings(jobs, config):
    """One paired endpoint per training seed; conditions are not replicates."""
    findings = {}
    for strategy in config["strategies"]:
        completed = [job for job in jobs.values() if job["strategy"] == strategy]
        rows, competence = {}, {}
        for job in completed:
            client = job["clients"][config["target_source"]]
            endpoint = client["rounds"][-1]
            for name, comparison in endpoint["paired_comparisons"].items():
                rows.setdefault(name, []).append({"seed": job["seed"], **comparison})
            for arm, curve in client["cold_target_curves"].items():
                competence.setdefault(arm, []).append({"seed": job["seed"], "crossings": curve["crossings"],
                    "competent_at_initial": curve["checkpoints"][0]["competent"],
                    "competent_at_endpoint": curve["checkpoints"][-1]["competent"]})
        paired = {}
        for name, comparisons in rows.items():
            differences = [row["treatment_minus_control_cost"] for row in comparisons]
            paired[name] = {"training_seed_pairs": len(comparisons),
                "lower_cost_seeds": sum(value < -1e-12 for value in differences),
                "tied_cost_seeds": sum(abs(value) <= 1e-12 for value in differences),
                "higher_cost_seeds": sum(value > 1e-12 for value in differences),
                "median_paired_cost_difference": float(np.median(differences)),
                "by_seed": comparisons}
        findings[strategy] = {"endpoint": paired, "competence_by_seed": competence,
            "inference_scope": "Descriptive paired training-seed results. No confidence interval or efficacy claim from deterministic conditions as extra replicates."}
    return findings


def analyze(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if (manifest["schema"] != 1 or manifest["study"] != "live_torax_critic_federation_cold_start_development"
            or not set(manifest["arms"]) <= set(TRAIN_ARMS)
            or not {"local", "reset_only", "raw_federated", "deployed", "joint_sac"} <= set(manifest["arms"])
            or len(set(manifest["arms"])) != len(manifest["arms"])
            or manifest["comparator"] != "calibration_fixed_command"):
        raise ValueError("Unsupported study/arm contract")
    provenance = audit_sources(directory, manifest)
    config = manifest["config"]
    state = read_json(directory / "study_state.json")
    summary = read_json(directory / "study_summary.json") if (directory / "study_summary.json").exists() else None
    if state["identity"] != manifest["identity"]:
        raise ValueError("Study state identity mismatch")
    if summary and (summary["identity"] != manifest["identity"] or summary["expected_budget"] != manifest["expected_budget"]):
        raise ValueError("Study summary identity or budget mismatch")
    jobs, incomplete, totals, paired_rows = {}, {}, {}, []
    expected_names = [f"{strategy}-seed-{seed}" for strategy in config["strategies"] for seed in config["seeds"]]
    if len(state["completed_jobs"]) != len(set(state["completed_jobs"])) or not set(state["completed_jobs"]) <= set(expected_names):
        raise ValueError("Unexpected or duplicate completed study jobs")
    for name in expected_names:
        job_dir = directory / "jobs" / name
        outcome = read_json(job_dir / "summary.json") if (job_dir / "summary.json").exists() else {}
        if outcome.get("status") != "complete" or name not in state["completed_jobs"]:
            incomplete[name] = {"status": outcome.get("status", "not_completed"), "error": outcome.get("error"),
                               "attempts": incomplete_attempts(job_dir, manifest["reference"], config["beta_mae_tolerance"])}
            continue
        job = read_json(job_dir / "job.json")
        if (outcome["job"] != job or outcome["completed_rounds"] != config["rounds"]
                or (outcome.get("recovered_from_acknowledgements") is not True
                    and outcome["process_exit_codes"] != [0] * (len(config["facility_factors"]) + 1))):
            raise ValueError("Completed job has failed processes or inconsistent identity")
        meta = audit_coordinator(job_dir / "coordinator.npz", cohort=job["cohort"],
                                 rounds=config["rounds"], strategy=job["strategy"])
        clipping = audit_job_contract(meta, job, manifest)
        if outcome["consumed_updates"] != {role: config["rounds"] * len(job["cohort"]) for role in ROLES}:
            raise ValueError("Completed job update accounting mismatch")
        clients = {}
        for source in job["cohort"]:
            client = audit_client(job_dir / "clients" / source, source, manifest, meta, job["seed"])
            clients[source] = client
            _add_work(totals, client["work"])
            for row in client["rounds"]:
                for comparison, measures in row["paired_comparisons"].items():
                    paired_rows.append({"strategy": job["strategy"], "seed": job["seed"], "source": source,
                                        "round": row["round"], "target_training_shots": row["additional_training_shots_per_arm"],
                                        "comparison": comparison, **measures})
        jobs[name] = {"strategy": job["strategy"], "seed": job["seed"], "clients": clients,
                      "clipping": clipping, "clipped_source_branch_updates": sum(row["active"] for row in clipping),
                      "coordinator_integrity_verified": True,
                      "recovered_from_acknowledgements": outcome.get("recovered_from_acknowledgements", False),
                      "process_exit_codes": outcome["process_exit_codes"],
                      "wall_seconds": outcome.get("wall_seconds"),
                      "recovery_wall_seconds": outcome.get("recovery_wall_seconds")}
    complete = (not incomplete and state["status"] == "complete" and state["pending_job"] is None
                and summary is not None and summary["status"] == "complete"
                and set(state["completed_jobs"]) == set(expected_names))
    episodes = sum(row["episodes"] for row in totals.values())
    windows = sum(row["windows"] for row in totals.values())
    if complete:
        for category in ("donor_pretraining", "initial_evaluation", "round_training", "gate", "reporting"):
            if totals.get(category, {}).get("episodes", 0) != manifest["expected_budget"][category + "_episodes"]:
                raise ValueError("Measured complete study work differs from frozen budget")
        if episodes != manifest["expected_budget"]["total_episodes"] or windows > manifest["expected_budget"]["maximum_action_windows"]:
            raise ValueError("Total completed study work differs from frozen budget")
    return {"schema": 1, "status": "complete_verified" if complete else "incomplete",
            "provenance": provenance, "expected_budget": manifest["expected_budget"],
            "completed_job_work": totals, "completed_job_episodes": episodes, "completed_job_windows": windows,
            "jobs": jobs, "incomplete_jobs": incomplete, "paired_by_seed_facility_round": paired_rows,
            "cold_target_findings": target_findings(jobs, config),
            "calibration_cost": {"reference_shot_windows": manifest["reference"].get("calibration_windows"),
                "scope": "Reference shot only; prior calibration sweeps and pilot development are additional and are not reconstructed from this study"},
            "interpretation": [
                "Primary comparison is cold_target raw_federated against paired local, reset_only and initial_base_pull; deployed fallback is reported separately.",
                "Only two training seeds by default; deterministic reporting conditions and facilities are not independent replicates.",
                "Zero-shot point precedes target learning and transfer; later points follow local learning plus transfer and repair. There is no immediate pre/post-transfer evaluation.",
                "Competence is an explicit development threshold on beta_N tracking AND Te profile error, not a reactor operational standard.",
                "First observed crossing is interval censored by checkpoint spacing; two consecutive passes do not guarantee future competence.",
                "Training-shot axis excludes donor pretraining, gate/evaluation shots and repair SGD; their actual work is counted separately.",
                "The fixed calibrated command is a comparator, not a trained controller; its curves use the experiment checkpoint index and consume zero training shots.",
                "Known compatible SPARC-like initial-temperature regimes do not establish transfer across different real reactor dynamics."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    output = args.output or args.directory / "independent_analysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "complete_jobs": len(result["jobs"]),
                      "incomplete_jobs": len(result["incomplete_jobs"]),
                      "episodes": result["completed_job_episodes"], "windows": result["completed_job_windows"],
                      "analysis": str(output)}, indent=2))


if __name__ == "__main__":
    main()
