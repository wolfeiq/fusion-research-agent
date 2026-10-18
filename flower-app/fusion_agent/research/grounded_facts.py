"""Deterministic, source-bound statements about captured shadow results.

These cards are descriptive projections, not model prose, an independent audit
of trajectories, or authorization to adopt a controller. Unknown/malformed
records are omitted. A valid source hash binds every card to the exact snapshot.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy

from .evidence import EvidenceError, resolve_pointer
from .shadow import CHANGES
from .shadow_analysis import ARMS, COMPARISON_SCHEMA

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,95}\Z")
_TERMINAL = {"completed", "failed", "timed_out", "interrupted"}
_ROLES = {"heating_td3": "heating TD3", "deposition_sac": "deposition SAC"}
# Registered shadow knobs have explicit paths in the recorded learner configs.
# Additional learner fields do not become publishable or adjustable implicitly.
_CONFIG_PATHS = {
    "td3_actor_lr": ("td3", "actor_lr"),
    "sac_actor_lr": ("sac", "actor_lr"),
    "sac_alpha_lr": ("sac", "alpha_lr"),
    "td3_mix": ("td3", "mix"),
    "sac_mix": ("sac", "mix"),
    "clipping_initial_norm_fraction": ("clipping_initial_norm_fraction",),
    "heating_critic_coordinates": ("td3", "critic_action_coordinates"),
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _configuration_value(config, parameter):
    """Read only the registered path and validate its actual captured value."""
    value = config
    for key in _CONFIG_PATHS[parameter]:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    spec = CHANGES[parameter]
    if spec["type"] == "number":
        return value if _number(value) and spec["min"] <= value <= spec["max"] else None
    if spec["type"] == "select":
        return value if isinstance(value, str) and value in spec["options"] else None
    return None


def _count(value):
    return type(value) is int and value >= 0


def _label(value):
    return (isinstance(value, str) and 1 <= len(value) <= 96
            and all(ord(char) >= 32 for char in value))


def _pointer(*parts):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _source(source):
    if (not isinstance(source, dict) or source.get("kind") != "experiment"
            or not isinstance(source.get("id"), str) or not _ID.fullmatch(source["id"])
            or not isinstance(source.get("metadata"), dict)):
        return None
    content = source.get("content", source.get("canonicalcontent"))
    try:
        serialized = content if isinstance(content, str) else _json(content)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if source.get("sha256") != digest:
            return None
        # resolve_pointer uses the strict duplicate-key/non-finite JSON parser.
        document = resolve_pointer(serialized, "")
        if not isinstance(document, dict):
            return None
        adapter = source["metadata"].get("adapter")
        if adapter == "shadow_comparison" and document.get("schema") == COMPARISON_SCHEMA:
            comparison, prefix = document, ""
            provenance = {"adapter": adapter, "origin": "saved_shadow_snapshot"}
        elif (adapter == "shadow_execution"
              and document.get("schema") == "fusion-advisor-shadow-evidence-v1"):
            comparison, prefix = document.get("comparison"), "/comparison"
            if comparison is None:
                # A real worker can fail before emitting its result. Preserve
                # that failure without synthesizing a comparison or metrics.
                if ("comparison" not in document or document.get("status") not in
                        {"failed", "timed_out", "interrupted"}):
                    return None
            elif not isinstance(comparison, dict) or comparison.get("schema") != COMPARISON_SCHEMA:
                return None
            provenance = {"adapter": adapter, "origin": "research_shadow_execution"}
        else:
            return None
        if comparison is not None and comparison.get("status") not in {"complete", "partial", "failed"}:
            return None
        # Origin is based on the registered adapter, never an E/D name guess.
        for key in ("scope", "verification_scope", "freshness", "input_source_id"):
            if isinstance(source["metadata"].get(key), str):
                provenance[key] = source["metadata"][key]
        trial_id = document.get("trial_id", (comparison or {}).get("provenance", {}).get("trial_id"))
        if _label(trial_id):
            provenance["trial_id"] = trial_id
        return document, comparison, prefix, provenance
    except (ValueError, TypeError, AttributeError, RecursionError, EvidenceError):
        return None


def _card(source, document, provenance, kind, text, pointers, context=None, derived=False):
    try:
        evidence = []
        for pointer in dict.fromkeys(pointers):
            quote = _json(resolve_pointer(document, pointer))
            if len(quote) > 2000 or len(pointer) > 512:
                return None
            evidence.append({"source_id": source["id"], "locator": pointer, "quote": quote})
        if not 1 <= len(evidence) <= 16:
            return None
        identity = {"source_id": source["id"], "source_sha256": source["sha256"],
                    "kind": kind, "text": text, "evidence": evidence}
        return {"id": "F" + hashlib.sha256(_json(identity).encode()).hexdigest()[:24],
                **identity, "fact_type": kind, "provenance": deepcopy(provenance),
                "context": deepcopy(context or {}), "derived": derived,
                "publication_basis": "deterministic_schema_projection"}
    except (ValueError, TypeError, EvidenceError):
        return None


def _complete(comparison, document, prefix):
    if (comparison.get("status") != "complete"
            or comparison.get("worker_reported_status") != "complete"
            or comparison.get("execution_status") != "completed"
            or comparison.get("failures") != []
            or (prefix and document.get("status") != "completed")):
        return False
    seeds = comparison.get("expected_training_seeds")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(not _count(seed) for seed in seeds) or len(set(seeds)) != len(seeds)):
        return False
    branches = comparison.get("branches")
    if not isinstance(branches, dict):
        return False
    for name in ("baseline", "candidate"):
        branch = branches.get(name, {})
        rows = branch.get("seeds", {})
        if (branch.get("status") != "complete" or not isinstance(rows, dict)
                or set(rows) != {str(seed) for seed in seeds}):
            return False
        for seed in seeds:
            row = rows[str(seed)]
            if (not isinstance(row, dict) or row.get("status") != "complete"
                    or type(row.get("seed")) is not int or row["seed"] != seed
                    or row.get("conditions_match") is not True
                    or not isinstance(row.get("arms"), dict)
                    or set(row["arms"]) != set(ARMS)):
                return False
    return True


def _valid_endpoint(row):
    if not isinstance(row, dict) or row.get("valid_complete") is not True:
        return False
    return (type(row.get("competent")) is bool and _number(row.get("mean_cost"))
            and _count(row.get("cases")) and row["cases"] > 0
            and all(_count(row.get(key)) for key in ("complete_cases", "cases_with_events", "cases_with_invalid_physics"))
            and row.get("complete_cases") == row["cases"]
            and row.get("cases_with_events") == 0
            and row.get("cases_with_invalid_physics") == 0)


def _valid_telemetry(row, comparison):
    if (not isinstance(row, dict) or row.get("status") != "recorded_scalars_consistent"
            or row.get("role") not in _ROLES or row.get("configuration") not in ("baseline", "candidate")
            or row.get("arm") != "clipped" or not _count(row.get("seed"))
            or row["seed"] not in comparison["expected_training_seeds"]
            or not _count(row.get("round")) or row["round"] < 1):
        return False
    seed_row = comparison["branches"][row["configuration"]]["seeds"][str(row["seed"])]
    if row["round"] not in [point.get("round") for point in seed_row.get("competence_checkpoints", [])
                            if isinstance(point, dict)]:
        return False
    identifiers = row.get("source_ids")
    if (not isinstance(identifiers, list) or not 1 <= len(identifiers) <= 16
            or not all(_label(item) for item in identifiers)
            or len(set(identifiers)) != len(identifiers) or row.get("source_count") != len(identifiers)):
        return False
    try:
        for key in ("source_norms", "clipping_factors"):
            if set(row[key]) != set(identifiers):
                return False
        for critic in ("q0", "q1"):
            threshold = row["thresholds"][critic]
            if not _number(threshold) or threshold <= 0:
                return False
            active = []
            for identifier in identifiers:
                factor = row["clipping_factors"][identifier][critic]
                norm = row["source_norms"][identifier][critic]
                if (not _number(factor) or not 0 < factor <= 1
                        or not _number(norm) or norm < 0
                        or not math.isclose(factor, min(1, threshold / norm) if norm else 1,
                                            rel_tol=1e-8, abs_tol=1e-12)):
                    return False
                if factor < 1:
                    active.append(identifier)
            if (sorted(active) != sorted(row["clipped_sources"][critic])
                    or type(row["clipped_source_counts"][critic]) is not int
                    or row["clipped_source_counts"][critic] != len(active)):
                return False
        hashes = row["proposal_hashes"]
        if not all(isinstance(hashes.get(key), str) and _HASH.fullmatch(hashes[key])
                   for key in ("uniform", "clipped")):
            return False
        return row.get("recorded_uniform_and_clipped_hashes_equal") is (hashes["uniform"] == hashes["clipped"])
    except (KeyError, TypeError, ValueError):
        return False


def _build(source, document, comparison, prefix, provenance):
    facts = []

    def add(kind, text, pointers, context=None, derived=False):
        card = _card(source, document, provenance, kind, text, pointers, context, derived)
        if card is not None:
            facts.append(card)

    if comparison is None:
        add("trial_status", f"This shadow execution records status {document['status']}; no comparison snapshot was produced.",
            ["/status", "/comparison"])
        if type(document.get("policy_activated")) is bool:
            add("policy_activation", f"The shadow execution records policy_activated={str(document['policy_activated']).lower()}.",
                ["/policy_activated"])
        return facts
    status = comparison["status"]
    execution = document.get("status") if prefix else comparison.get("execution_status")
    if execution in _TERMINAL:
        add("trial_status", f"This shadow source records execution status {execution} and comparison status {status}.",
            ["/status" if prefix else "/execution_status", prefix + "/status"])
    if prefix and type(document.get("policy_activated")) is bool:
        add("policy_activation", f"The shadow execution records policy_activated={str(document['policy_activated']).lower()}.",
            ["/policy_activated"])
    # Failed/incomplete runs cannot produce inactivity, successful pairing or
    # complete-cost claims just because some plausible numbers survived.
    if not _complete(comparison, document, prefix):
        return facts
    branches = comparison["branches"]
    for parameter, config_path in _CONFIG_PATHS.items():
        values = {name: _configuration_value(branches[name].get("configs"), parameter)
                  for name in ("baseline", "candidate")}
        paths = {name: prefix + _pointer("branches", name, "configs", *config_path)
                 for name in values}
        if all(value is not None for value in values.values()) and values["baseline"] != values["candidate"]:
            add("configuration_change",
                f"The recorded {CHANGES[parameter]['label']} ({parameter}) changes from "
                f"baseline {_json(values['baseline'])} to candidate {_json(values['candidate'])}.",
                [paths["baseline"], paths["candidate"]],
                {"parameter": parameter, "comparison": "baseline_to_candidate",
                 "baseline": values["baseline"], "candidate": values["candidate"]})
        for name, value in values.items():
            if value is not None:
                add("configuration", f"The {name} configuration records {parameter}={_json(value)}.",
                    [paths[name]], {"configuration": name, "parameter": parameter})
    totals, accounting_pointers = {}, []
    for metric in ("training_episodes", "evaluation_episodes", "windows"):
        values = [branches[name].get("accounting", {}).get(metric) for name in ("baseline", "candidate")]
        if not all(_count(value) for value in values):
            break
        totals[metric] = sum(values)
        accounting_pointers.extend(prefix + _pointer("branches", name, "accounting", metric)
                                   for name in ("baseline", "candidate"))
    if len(totals) == 3:
        add("actual_budget", "Summing the two configurations' recorded accounting gives "
            f"{totals['training_episodes']} training episodes plus {totals['evaluation_episodes']} evaluation episodes "
            f"({totals['training_episodes'] + totals['evaluation_episodes']} total), with {totals['windows']} action windows. "
            "These are executed episode totals, not optimizer-update counts; prior calibration is not included in this sum.",
            accounting_pointers, derived=True)
    for arm in ARMS:
        rows = comparison.get("candidate_vs_baseline_raw_endpoint", {}).get(arm, {}).get("by_seed", [])
        if not isinstance(rows, list) or len(rows) > 3:
            continue
        seed_counts = {}
        for row in rows:
            if isinstance(row, dict) and _count(row.get("seed")):
                seed_counts[row["seed"]] = seed_counts.get(row["seed"], 0) + 1
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or row.get("comparable") is not True
                    or not _count(row.get("seed")) or row["seed"] not in comparison["expected_training_seeds"]
                    or seed_counts.get(row["seed"]) != 1):
                continue
            seed = row["seed"]
            baseline, candidate = [branches[name]["seeds"][str(seed)]["arms"][arm].get("raw")
                                   for name in ("baseline", "candidate")]
            if not all(_valid_endpoint(value) for value in (baseline, candidate)):
                continue
            difference = row.get("mean_cost_difference")
            if (not _number(difference)
                    or difference != candidate["mean_cost"] - baseline["mean_cost"]
                    or baseline["cases"] != candidate["cases"]
                    or row.get("baseline_competent") is not baseline["competent"]
                    or row.get("candidate_competent") is not candidate["competent"]
                    or not isinstance(baseline.get("condition_signature"), str)
                    or not _HASH.fullmatch(baseline["condition_signature"])
                    or baseline["condition_signature"] != candidate.get("condition_signature")):
                continue
            path = prefix + _pointer("candidate_vs_baseline_raw_endpoint", arm, "by_seed", index)
            context = {"seed": seed, "arm": arm, "outcome": "raw_endpoint"}
            add("paired_raw_endpoint", f"For training seed {seed}, the {arm} arm's paired raw endpoint mean-cost difference "
                f"(candidate minus baseline) is {difference!r}. This is an endpoint cost difference, not a shots-to-competence measurement.",
                [path + "/seed", path + "/comparable", path + "/mean_cost_difference"], context)
            add("paired_competence", f"For training seed {seed}, the {arm} arm's raw endpoint competence flags are "
                f"baseline={str(baseline['competent']).lower()} and candidate={str(candidate['competent']).lower()}.",
                [path + "/seed", path + "/baseline_competent", path + "/candidate_competent"], context)
    for name in ("baseline", "candidate"):
        for seed in comparison["expected_training_seeds"]:
            for arm in ("reset_only", "uniform", "clipped"):
                row = branches[name]["seeds"][str(seed)]["arms"][arm]
                if not isinstance(row, dict) or not isinstance(row.get("gate"), dict):
                    continue
                gate = row["gate"]
                if (type(gate.get("accept")) is not bool or not _valid_endpoint(row.get("raw"))
                        or not _valid_endpoint(row.get("deployed"))
                        or (gate["accept"] and row["raw"] != row["deployed"])):
                    continue
                path = prefix + _pointer("branches", name, "seeds", str(seed), "arms", arm)
                decision = "accepted" if gate["accept"] else "rejected"
                add("gate_outcome", f"For {name}, training seed {seed}, {arm}, the gate {decision} the proposal. "
                    f"The recorded raw endpoint mean cost is {row['raw']['mean_cost']!r}; the separately recorded "
                    f"gate-selected deployed endpoint mean cost is {row['deployed']['mean_cost']!r}. "
                    "The gate decision does not turn the deployed diagnostic into evidence of raw learning.",
                    [path + "/gate/accept", path + "/raw/mean_cost", path + "/deployed/mean_cost",
                     prefix + _pointer("branches", name, "seeds", str(seed), "seed")],
                    {"configuration": name, "seed": seed, "arm": arm, "outcome": "gate_selected_endpoint"})
    diagnostics = document.get("round_aggregation_diagnostics", {})
    if (isinstance(diagnostics, dict) and diagnostics.get("schema") == "fusion-shadow-clipping-diagnostic-v1"
            and isinstance(diagnostics.get("raw_result_sha256"), str)
            and _HASH.fullmatch(diagnostics["raw_result_sha256"])
            and diagnostics["raw_result_sha256"] == comparison.get("provenance", {}).get("result_sha256")
            and isinstance(diagnostics.get("records"), list) and len(diagnostics["records"]) <= 24):
        identities = {}
        for row in diagnostics["records"]:
            if isinstance(row, dict):
                key = _json({name: row.get(name) for name in ("configuration", "seed", "round", "role", "arm")})
                identities[key] = identities.get(key, 0) + 1
        for index, row in enumerate(diagnostics["records"]):
            if not _valid_telemetry(row, comparison):
                continue
            context = {key: row[key] for key in ("configuration", "seed", "round", "role", "arm")}
            if identities[_json(context)] != 1:
                continue
            for critic in ("q0", "q1"):
                factors = ", ".join(f"{name}={row['clipping_factors'][name][critic]!r}" for name in row["source_ids"])
                add("clipping", f"For {row['configuration']}, training seed {row['seed']}, round {row['round']}, "
                    f"the clipped arm's {_ROLES[row['role']]} {critic} critic clipped "
                    f"{row['clipped_source_counts'][critic]} of {row['source_count']} recorded contributors. "
                    f"Recorded clipping factors: {factors}.",
                    [_pointer("round_aggregation_diagnostics", "records", index)], {**context, "critic": critic})
    return facts


def source_provenance(source: dict) -> dict | None:
    """Return recorded provenance for a recognized, digest-verified source."""
    parsed = _source(source)
    return deepcopy(parsed[3]) if parsed is not None else None


def build_grounded_facts(sources: dict | list) -> list[dict]:
    """Return deterministic fact cards from recognized, hash-bound snapshots.

    Accept a source registry, a list of snapshots, or one source. No filesystem,
    network, model, optimizer or current experiment state is accessed. Every
    citation quotes an exact value in its bound source; no missing fields are
    repaired or guessed. Repeated source identities with different content are
    all rejected. Duplicate identical snapshots are deduplicated.
    """
    if isinstance(sources, dict):
        sources = [sources] if "content" in sources or "canonicalcontent" in sources else list(sources.values())
    if not isinstance(sources, list):
        return []
    identity_hashes = {}
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("id"), str):
            identity_hashes.setdefault(source["id"], set()).add(str(source.get("sha256")))
    cards = {}
    for source in sources:
        parsed = _source(source)
        if parsed is None or len(identity_hashes[source["id"]]) != 1:
            continue
        try:
            for card in _build(source, *parsed):
                cards[card["id"]] = card
        except (TypeError, ValueError, KeyError, AttributeError, RecursionError):
            # No partial positive claims survive a structurally broken source.
            continue
    return list(cards.values())
