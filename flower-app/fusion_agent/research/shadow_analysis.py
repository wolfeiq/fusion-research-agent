"""Deterministic paired analysis of isolated TORAX shadow-worker results.

This module imports no learner and never launches a trial or installs a policy.
Training seeds are the replicate unit; reporting conditions are not extra seeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from typing import Any

RESULT_SCHEMA = "fusion-shadow-worker-result-v1"
COMPARISON_SCHEMA = "fusion-shadow-comparison-v1"
ARMS = ("local", "reset_only", "uniform", "clipped")
METRICS = ("mean_cost", "mean_beta_mae", "mean_Te_profile_rmse_keV")
STATUSES = {"complete", "partial", "failed"}
RECORDED_STATUSES = STATUSES | {"pending", "running"}
MAX_SEEDS = 3
DEVELOPMENT_SEED_SEQUENCE = (17, 29, 43, 59, 71)
TRIAL_FIELDS = {
    "td3_actor_lr", "sac_actor_lr", "sac_alpha_lr", "td3_mix", "sac_mix",
    "clipping_initial_norm_fraction", "heating_critic_coordinates",
}


class ShadowAnalysisError(ValueError):
    """The recorded worker result violates the shadow evidence contract."""


def canonical_sha256(value: Any) -> str:
    """Hash the exact canonical JSON representation used by this adapter."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ShadowAnalysisError(f"{name} must be an object")
    return value


def _number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowAnalysisError(f"{name} must be finite numeric data or null")
    if not math.isfinite(value):
        raise ShadowAnalysisError(f"{name} must be finite numeric data or null")
    return float(value)


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ShadowAnalysisError(f"{name} must be a nonnegative integer")
    return value


def _summary(value: Any, name: str) -> dict:
    value = _mapping(value, name)
    result = {metric: _number(value.get(metric), f"{name}.{metric}")
              for metric in METRICS}
    for key in ("cases", "complete_cases", "cases_with_events",
                "cases_with_invalid_physics"):
        result[key] = _count(value.get(key), f"{name}.{key}")
    if any(result[key] > result["cases"] for key in (
        "complete_cases", "cases_with_events", "cases_with_invalid_physics"
    )):
        raise ShadowAnalysisError(f"{name} case counts are inconsistent")
    if type(value.get("competent")) is not bool:
        raise ShadowAnalysisError(f"{name}.competent must be a boolean")
    result["competent"] = value["competent"]
    if "per_case" in value:
        cases = value["per_case"]
        if not isinstance(cases, list) or len(cases) != result["cases"] or len(cases) > 32:
            raise ShadowAnalysisError(f"{name} per-case condition matrix is inconsistent")
        if any(not isinstance(case, dict) or "seed" not in case or "scenario" not in case for case in cases):
            raise ShadowAnalysisError(f"{name} is missing per-case seed/scenario identity")
        result["condition_signature"] = canonical_sha256([
            {"seed": case["seed"], "scenario": case["scenario"]} for case in cases
        ])
    result["valid_complete"] = bool(
        result["cases"] > 0
        and result["complete_cases"] == result["cases"]
        and not result["cases_with_events"]
        and not result["cases_with_invalid_physics"]
        and all(result[metric] is not None for metric in METRICS)
    )
    if result["competent"] and not result["valid_complete"]:
        raise ShadowAnalysisError(f"{name} claims competence despite invalid/incomplete cases")
    return result


def _arms(value: Any, name: str, *, complete: bool = True) -> dict:
    value = _mapping(value, name)
    if not set(value) <= set(ARMS) or (complete and set(value) != set(ARMS)):
        raise ShadowAnalysisError(f"{name} must preserve all four arms")
    result = {}
    for arm in ARMS:
        if arm not in value:
            continue
        row = _mapping(value[arm], f"{name}.{arm}")
        gate = row.get("gate")
        if gate is not None and not isinstance(gate, dict):
            raise ShadowAnalysisError("Gate evidence must be an object or null")
        # Keep the gate's recorded decision, never reinterpret it as raw learning.
        result[arm] = {
            "raw": _summary(row.get("raw"), f"{name}.{arm}.raw"),
            "deployed": (_summary(row["deployed"], f"{name}.{arm}.deployed")
                         if row.get("deployed") is not None else None),
            "gate": deepcopy(gate),
        }
    return result


def _delta(candidate: dict, baseline: dict) -> dict:
    """Only compare complete matched endpoint case matrices."""
    comparable = (
        candidate["valid_complete"] and baseline["valid_complete"]
        and candidate["cases"] == baseline["cases"]
        and candidate.get("condition_signature") == baseline.get("condition_signature")
    )
    result = {"comparable": comparable}
    for metric in METRICS:
        result[metric + "_difference"] = (
            candidate[metric] - baseline[metric] if comparable else None
        )
    result["relative_cost_change"] = (
        (candidate["mean_cost"] - baseline["mean_cost"]) / abs(baseline["mean_cost"])
        if comparable and baseline["mean_cost"] != 0 else None
    )
    result["baseline_competent"] = baseline["competent"]
    result["candidate_competent"] = candidate["competent"]
    return result


def _aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["comparable"]]
    differences = [row["mean_cost_difference"] for row in valid]
    return {
        "paired_training_seeds": len(valid),
        "lower_cost_seeds": sum(value < -1e-12 for value in differences),
        "tied_cost_seeds": sum(abs(value) <= 1e-12 for value in differences),
        "higher_cost_seeds": sum(value > 1e-12 for value in differences),
        "median_paired_cost_difference": (
            statistics.median(differences) if differences else None
        ),
        "by_seed": rows,
    }


def _accounting(value: Any) -> dict:
    """Keep measured totals; compact verbose optimizer ledgers transparently."""
    result = deepcopy(_mapping(value, "Cost accounting"))
    for kind in ("contribution_optimization", "repair_optimization"):
        if kind not in result:
            continue
        rows = result[kind]
        if not isinstance(rows, list) or len(rows) > 200:
            raise ShadowAnalysisError("Optimizer accounting ledger exceeds its bound")
        roles = {}
        for row in rows:
            row = _mapping(row, "Optimizer ledger row")
            if kind == "repair_optimization":
                entries = _mapping(row.get("roles"), "Repair role accounting")
            else:
                role = row.get("role")
                if role not in ("heating_td3", "deposition_sac"):
                    raise ShadowAnalysisError("Unknown contribution role")
                entries = {role: {key: item for key, item in row.items()
                                  if key not in {"arm", "role", "round", "seed", "source"}}}
            for role, values in entries.items():
                if role not in ("heating_td3", "deposition_sac"):
                    raise ShadowAnalysisError("Unknown repair role")
                totals = roles.setdefault(role, {})
                for key, count in _mapping(values, "Role accounting").items():
                    totals[key] = totals.get(key, 0) + _count(count, key)
        result[kind] = {
            "record_count": len(rows), "summed_by_role": roles,
            "projection": "Summed across recorded sources, arms, rounds and seeds; original per-record ledger remains in the hash-bound worker result.",
        }
    result["source_target_split"] = (
        "Recorded episode categories retained below"
        if "episode_categories" in result
        else "Training episode total combines donor and target work; no separate actual category totals were exported."
    )
    return result


def analyze_shadow_result(result: dict) -> dict:
    """Produce a bounded descriptive comparison from the worker's saved result.

    The result must carry frozen, shared condition hashes and expected seeds.
    Invalid schema is rejected. Failed/missing trials remain visible and block
    a positive recommendation rather than being discarded from the denominator.
    """
    result = _mapping(result, "Worker result")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") not in RECORDED_STATUSES:
        raise ShadowAnalysisError("Unsupported shadow worker schema or status")
    # Validate all JSON, including provenance/accounting, before copying fields.
    digest = canonical_sha256(result)
    config = _mapping(result.get("config"), "Worker config")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_SEEDS
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ShadowAnalysisError("Shadow config requires one to three unique training seeds")
    conditions = _mapping(result.get("conditions"), "Shared conditions")
    conditions_hash = result.get("conditions_sha256")
    if conditions_hash != canonical_sha256(conditions):
        raise ShadowAnalysisError("Shared conditions hash does not match their content")
    recorded_failures = result.get("failures", [])
    if not isinstance(recorded_failures, list) or len(recorded_failures) > 30:
        raise ShadowAnalysisError("Worker failures must be a bounded list")
    failures = deepcopy(recorded_failures)
    branches, reasons, invalid_cases = {}, [], False
    for name in ("baseline", "candidate"):
        branch = _mapping(result.get(name), name)
        if branch.get("status") not in RECORDED_STATUSES:
            raise ShadowAnalysisError(f"{name} requires a recorded execution status")
        seed_rows = branch.get("seeds")
        if not isinstance(seed_rows, list) or len(seed_rows) > MAX_SEEDS:
            raise ShadowAnalysisError(f"{name}.seeds must be bounded")
        normalized = {}
        for row in seed_rows:
            row = _mapping(row, f"{name}.seed")
            seed = row.get("seed")
            if type(seed) is not int or seed not in seeds or seed in normalized:
                raise ShadowAnalysisError("Unexpected or duplicate training seed")
            status = row.get("status")
            if status not in RECORDED_STATUSES:
                raise ShadowAnalysisError("Every training seed requires a status")
            matching = row.get("conditions_sha256") == conditions_hash
            if not matching:
                reasons.append(f"{name} seed {seed}: frozen conditions do not match")
            stable = row.get("training_stable")
            if stable is not None and type(stable) is not bool:
                raise ShadowAnalysisError("training_stable must be a boolean or null")
            selected = {
                "seed": seed, "status": status, "conditions_match": matching,
                "training_stable": stable, "initial": None, "arms": {},
                "competence_checkpoints": [],
            }
            if row.get("initial") is not None:
                selected["initial"] = _summary(row["initial"], f"{name}.initial")
                invalid_cases |= not selected["initial"]["valid_complete"]
            if row.get("arms"):
                selected["arms"] = _arms(row["arms"], f"{name}.arms", complete=status == "complete")
                invalid_cases |= any(
                    not arm["raw"]["valid_complete"]
                    for arm in selected["arms"].values()
                )
            elif status == "complete":
                raise ShadowAnalysisError("Completed training seed is missing endpoint arms")
            rounds = row.get("rounds", [])
            if not isinstance(rounds, list) or len(rounds) > 2:
                raise ShadowAnalysisError("Shadow rounds must be a bounded list")
            last_round, last_shots = 0, 0
            for point in rounds:
                point = _mapping(point, "Round evidence")
                round_number = _count(point.get("round"), "round")
                shots = _count(point.get("target_training_shots_per_arm"), "shots")
                if round_number <= last_round or shots <= last_shots:
                    raise ShadowAnalysisError("Round and training-shot indices must increase")
                point_arms = _arms(point.get("arms"), "Round arms", complete=status == "complete")
                selected["competence_checkpoints"].append({
                    "round": round_number, "target_training_shots_per_arm": shots,
                    "raw_competence": {arm: point_arms[arm]["raw"]["competent"]
                                       for arm in point_arms},
                })
                last_round, last_shots = round_number, shots
            if status == "complete" and rounds and selected["arms"] != point_arms:
                raise ShadowAnalysisError("Endpoint summaries disagree with the last saved round")
            if status == "complete" and len(rounds) != config.get("rounds"):
                raise ShadowAnalysisError("Completed seed does not contain the configured rounds")
            normalized[seed] = selected
        for seed in seeds:
            if seed not in normalized:
                failures.append({"branch": name, "seed": seed, "status": "missing"})
        branches[name] = {
            "status": branch["status"], "configs": deepcopy(branch.get("configs", {})),
            "accounting": _accounting(branch.get("accounting", {})),
            "seeds": normalized,
        }
    complete = (
        result["status"] == "complete" and not failures
        and all(branch["status"] == "complete" and len(branch["seeds"]) == len(seeds)
                and all(row["status"] == "complete" for row in branch["seeds"].values())
                for branch in branches.values())
    )
    stable = all(
        row["training_stable"] is True
        for branch in branches.values() for row in branch["seeds"].values()
    )
    paired = {arm: [] for arm in ARMS}
    within = {name: {f"{fed}_vs_{control}": [] for fed in ("uniform", "clipped")
                    for control in ("local", "reset_only")} for name in branches}
    for seed in seeds:
        baseline, candidate = (branches[name]["seeds"].get(seed)
                               for name in ("baseline", "candidate"))
        if (baseline and candidate and baseline["arms"] and candidate["arms"]
                and baseline["conditions_match"] and candidate["conditions_match"]
                and baseline["status"] == candidate["status"] == "complete"):
            for arm in ARMS:
                raw = _delta(candidate["arms"][arm]["raw"], baseline["arms"][arm]["raw"])
                deployed = None
                if candidate["arms"][arm]["deployed"] and baseline["arms"][arm]["deployed"]:
                    deployed = _delta(candidate["arms"][arm]["deployed"], baseline["arms"][arm]["deployed"])
                paired[arm].append({"seed": seed, **raw, "deployed_comparison": deployed})
                if not raw["comparable"]:
                    reasons.append(f"{arm} seed {seed}: endpoint cases are not comparable")
        for name, branch in branches.items():
            row = branch["seeds"].get(seed)
            if not row or not row["arms"] or not row["conditions_match"] or row["status"] != "complete":
                continue
            for fed in ("uniform", "clipped"):
                for control in ("local", "reset_only"):
                    within[name][f"{fed}_vs_{control}"].append({
                        "seed": seed, **_delta(row["arms"][fed]["raw"], row["arms"][control]["raw"]),
                    })
    paired = {name: _aggregate(rows) for name, rows in paired.items()}
    within = {name: {key: _aggregate(rows) for key, rows in comparisons.items()}
              for name, comparisons in within.items()}
    recommendation = _recommend(complete, stable, invalid_cases, reasons, paired)
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "complete" if complete else "failed" if result["status"] == "failed" else "partial",
        "worker_reported_status": result["status"],
        "worker_result_sha256": digest,
        "conditions_sha256": conditions_hash,
        "expected_training_seeds": seeds,
        "frozen_budget": deepcopy(conditions.get("budget", {})),
        "fairness": {"same_recorded_conditions": not reasons, "reasons": reasons,
                     "independent_replicate_unit": "training seed",
                     "condition_hash_scope": "Worker-recorded conditions, not a fresh raw trajectory audit"},
        "branches": branches,
        "candidate_vs_baseline_raw_endpoint": paired,
        "within_branch_raw_federation": within,
        "failures": failures,
        "recommendation": recommendation,
        "next_run_proposal": _next_run_proposal(config, complete, recommendation),
        "limitations": [
            "Development shadow experiment only; no reactor deployment or automatic policy adoption.",
            "One training seed cannot establish a federation benefit; conditions are not independent replicates.",
            "Endpoint cost, beta_N tracking MAE and Te temperature-profile RMSE measure different outcomes.",
            "Initial and one or two round checkpoints do not establish a shots-to-competence speedup.",
            "Gate-selected deployed results are diagnostic fallback outcomes; raw learning continues separately.",
            "Source training, target training, gate/evaluation work and optimizer updates must be counted separately.",
        ],
    }


def _recommend(complete, stable, invalid_cases, reasons, paired):
    """Conservative next-step recommendation, never policy activation."""
    if not complete or not stable or invalid_cases:
        code = "diagnose_before_comparison"
        text = "Preserve failures and diagnose incomplete, invalid or unstable learning before a larger federation trial."
    elif reasons or any(row["paired_training_seeds"] == 0 for row in paired.values()):
        code = "repair_comparison_contract"
        text = "Restore identical frozen conditions and paired training seeds before comparing endpoints."
    else:
        compared_rows = [item for arm in ARMS
                         for item in paired[arm]["by_seed"] if item["comparable"]]
        worse = any(
            any(item[metric + "_difference"] > 1e-12 for metric in METRICS)
            or (item["baseline_competent"] and not item["candidate_competent"])
            for item in compared_rows
        )
        better = any(item["mean_cost_difference"] < -1e-12 for item in compared_rows)
        if worse:
            code = "keep_baseline"
            text = "Keep the baseline for now; the candidate worsened cost, beta_N error or profile error in at least one paired raw arm, or lost competence. Diagnose this trade-off before revising."
        elif better:
            code = "repeat_varied_paired_seeds"
            text = "The candidate has lower observed raw endpoint cost in at least one arm of this development check. Freeze it and repeat across additional paired training seeds; this is not a federation efficacy or cold-start speedup claim."
        else:
            code = "inconclusive"
            text = "No consistent endpoint advantage is established. Preserve the baseline and test a separately justified, frozen hypothesis."
    return {"code": code, "text": text, "automatic_launch": False,
            "automatic_apply": False, "efficacy_claim": False}


def _next_run_proposal(config, complete, recommendation):
    """Replicate the same frozen hypothesis; do not select a winning new knob."""
    proposal = {
        "status": "proposal_only", "request": None, "budget": None,
        "automatic_launch": False, "automatic_apply": False,
    }
    if not complete or recommendation["code"] in {
        "diagnose_before_comparison", "repair_comparison_contract"
    }:
        proposal["reason"] = "Diagnose failure, instability or mismatched conditions before requesting another experiment."
        return proposal
    if config.get("change") not in TRIAL_FIELDS or "value" not in config:
        proposal["reason"] = "The saved result lacks a registered frozen candidate change; no executable request can be proposed."
        return proposal
    seeds = [seed for seed in DEVELOPMENT_SEED_SEQUENCE if seed not in config["seeds"]][:2]
    if len(seeds) != 2:
        proposal["reason"] = "Insufficient unused seeds in the registered development sequence; review the development ledger."
        return proposal
    rounds, count = config["rounds"], len(seeds)
    proposal.update(
        request={"schema": "fusion-shadow-config-v1", "change": config["change"],
                 "value": deepcopy(config["value"]), "seeds": seeds, "rounds": rounds},
        budget={
            "configurations": 2, "training_episodes": count * (16 + 16 * rounds),
            "donor_training_episodes": 16 * count,
            "target_training_episodes": 16 * count * rounds,
            "evaluation_episodes": count * (4 + 20 * rounds),
            "initial_and_fixed_evaluation_episodes": 4 * count,
            "gate_episodes": 8 * count * rounds, "report_episodes": 12 * count * rounds,
            "total_episodes": count * (20 + 36 * rounds),
            "maximum_windows": count * (20 + 36 * rounds) * 30,
            "maximum_wall_seconds": 600, "cpu_threads": 1,
            "calibration_scope": "Existing frozen reference reused; prior calibration work excluded.",
            "optimizer_scope": "Log contribution, head-repair, adaptation and entropy optimizer work separately; episode counts are not optimizer-work counts.",
        },
        reason="Repeat the identical baseline/candidate hypothesis across two additional paired development seeds, preserving null and negative outcomes. Keep the baseline meanwhile; do not change a knob to manufacture a federation win.",
        seed_selection_scope="Unused by this saved trial only. Check the development/confirmation ledger before launch; this proposal does not scan other runs.",
        invariants=[
            "Same source revision, calibration, candidate change/value, paired physical conditions, donor budgets, reporting matrix and local/reset/uniform/clipped controls.",
            "Same round count; distinguish raw outcomes from gate fallback and endpoint metrics from shots to competence.",
            "Create a new frozen manifest and retain this original attempt; no launch or installation is authorized by the recommendation.",
        ],
    )
    return proposal


def attach_execution_status(comparison: dict, execution_status: str) -> dict:
    """Bind analysis to process outcome without erasing its measured evidence."""
    if execution_status not in {"completed", "failed", "timed_out", "interrupted"}:
        raise ShadowAnalysisError("Shadow execution status must be terminal")
    if comparison.get("schema") != COMPARISON_SCHEMA:
        raise ShadowAnalysisError("Unsupported shadow comparison schema")
    result = deepcopy(comparison)
    result["execution_status"] = execution_status
    if execution_status == "completed":
        if result.get("status") != "complete":
            raise ShadowAnalysisError("Completed process lacks a complete comparison")
    else:
        result["status"] = "failed" if execution_status == "failed" else "partial"
        result["recommendation"] = {
            "code": "diagnose_before_comparison",
            "text": "The worker did not finish successfully. Preserve its measured and partial evidence, investigate the execution failure, and rerun the frozen comparison before drawing an advantage claim.",
            "automatic_launch": False, "automatic_apply": False,
            "efficacy_claim": False,
        }
        result["next_run_proposal"] = {
            "status": "proposal_only", "request": None, "budget": None,
            "reason": "Diagnose the failed, interrupted or timed-out worker before requesting another experiment.",
            "automatic_launch": False, "automatic_apply": False,
        }
    return result
