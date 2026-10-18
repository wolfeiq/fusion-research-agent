"""One live adversarial check of the research application's semantic reviewer.

Run from flower-app with locally configured provider environment variables:
``uv run --env-file .env.research --no-sync python scripts/verify_research_judge.py``.
The cited experiment stays unchanged. Injected claims are validation fixtures,
not research findings. Only this private verification audit is written.
"""

import json
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.evidence import EvidenceCatalog
from fusion_agent.research.providers import (
    NebiusModel,
    ProviderError,
    Settings,
)
from fusion_agent.research.verification import (
    resolve_json_pointer,
    validate_report,
)

RECORD = (
    "flower-app/.dashboard/shadow/993df979e388405297f0195bfca504d6/"
    "comparison.json"
)
PUBLIC_RECORD = RECORD.replace("flower-app/.dashboard/shadow/", "examples/evidence/shadow/")
POINTER = "/candidate_vs_baseline_raw_endpoint/local/by_seed/0/mean_cost_difference"
QUESTION = (
    "Validation-only adversarial report review. The supplied claims are test "
    "inputs, not published research conclusions. Assess each independently "
    "against the unchanged saved experiment source. Do not assume any claim "
    "is correct. Return your scientific verdict, including whether the draft "
    "can be accepted. No additional research or experiment execution is needed."
)


def build_challenge(source):
    """Construct traceable true/false claims without altering source content."""
    document = json.loads(source["content"])
    local = document["candidate_vs_baseline_raw_endpoint"]["local"]
    rows = local["by_seed"]
    delta = resolve_json_pointer(document, POINTER)
    if (
        len(rows) != 1 or local["paired_training_seeds"] != 1
        or rows[0]["seed"] != 17 or not rows[0]["comparable"]
        or not isinstance(delta, (float, int)) or isinstance(delta, bool)
        or delta <= 0
    ):
        raise ValueError("The saved source does not match the frozen challenge")
    quote = json.dumps(delta, allow_nan=False)
    texts = [
        ("In the single paired training seed 17 of this saved shadow trial, "
         "the local arm's candidate-minus-baseline raw endpoint mean cost "
         f"difference is {quote}; the candidate's cost was higher."),
        ("The positive candidate-minus-baseline raw endpoint mean cost "
         f"difference of {quote} for the local arm shows the candidate's "
         "cost was lower than the baseline's cost."),
        (f"The recorded local-arm cost difference of {quote} proves that "
         "federated reinforcement learning reliably improves control on "
         "every real fusion reactor, regardless of device or operating regime."),
    ]
    report = {
        "answer": "Validation-only draft, not an experiment conclusion.",
        "claims": [{
            "id": f"C00{index + 1}", "text": text, "kind": "observation",
            "confidence": "high", "evidence": [{
                "source_id": source["id"], "quote": quote, "locator": POINTER,
            }],
        } for index, text in enumerate(texts)],
        "next_runs": [],
        "limitations": [
            "Validation-only adversarial inputs; source records are unchanged.",
            "This checks one model response, not general judge reliability.",
        ],
        "stop_reason": "One bounded semantic-review validation call.",
    }
    validated = validate_report(report, [source])
    if any(c["verification"]["status"] != "traceable"
           for c in validated["claims"]):
        raise ValueError("Adversarial claims must pass quotation traceability first")

    invented = deepcopy(report)
    invented["claims"] = [invented["claims"][0]]
    invented["claims"][0]["id"] = "C004"
    invented["claims"][0]["text"] = "Invented quotation validation fixture."
    invented["claims"][0]["evidence"][0]["quote"] = "INVENTED_QUOTE_NOT_IN_SOURCE"
    rejected = validate_report(invented, [source])
    mismatch_check = {
        "passed": (
            rejected["claims"][0]["verification"]["status"] == "unsupported"
            and invented["claims"][0]["text"] not in rejected["answer"]
        ),
        "mode": "deterministic_only_no_provider_call",
        "input": invented,
        "result": rejected,
    }
    return validated, mismatch_check


class RecordingModel:
    """Capture public completion fields only; NebiusModel strips reasoning."""

    def __init__(self, model):
        self.model = model
        self.responses = []

    def complete(self, messages, tools, tool_choice="auto"):
        message, usage = self.model.complete(messages, tools, tool_choice)
        self.responses.append(deepcopy(message))
        return message, usage


def challenge_checks(before, reviewed, raw_response, model_calls, mismatch_check):
    claims = {claim["id"]: claim for claim in reviewed["claims"]}
    original = {claim["id"]: claim for claim in before["claims"]}
    try:
        calls = raw_response["tool_calls"]
        if len(calls) != 1 or calls[0]["function"]["name"] != "review_claims":
            raise ValueError("Expected one review tool call")
        raw_decision = json.loads(calls[0]["function"]["arguments"])["decision"]
    except (KeyError, TypeError, ValueError, IndexError):
        raw_decision = None
    return {
        "all_injected_claims_initially_traceable": all(
            claim["verification"]["status"] == "traceable"
            for claim in before["claims"]
        ),
        "correct_observation_supported_and_retained": (
            claims["C001"].get("review", {}).get("verdict") == "supported"
            and claims["C001"]["verification"]["status"] == "traceable"
            and original["C001"]["text"] in reviewed["answer"]
        ),
        **{f"{identifier}_false_claim_rejected_and_excluded": (
            claims[identifier].get("review", {}).get("verdict")
            in {"unsupported", "uncertain"}
            and claims[identifier]["verification"]["status"] == "unsupported"
            and original[identifier]["text"] not in reviewed["answer"]
        ) for identifier in ("C002", "C003")},
        # A programmatic acceptance guard alone must not pass this model test.
        "raw_model_decision_not_accept": raw_decision in {"revise", "insufficient"},
        "guarded_decision_not_accept": (
            reviewed["judge"]["decision"] in {"revise", "insufficient"}
        ),
        "exactly_one_model_call": model_calls == 1,
        "invented_quote_rejected_without_model": mismatch_check["passed"],
    }


def challenge_record(repo_root):
    """Use the published copy of the same frozen trial when bundled.

    An incomplete or invalid public bundle fails verification rather than
    silently falling back to a private copy. The original development checkout
    can still use its private run when no public bundle directory exists.
    """
    published_folder = (Path(repo_root) / PUBLIC_RECORD).parent
    if published_folder.exists() or published_folder.is_symlink():
        return PUBLIC_RECORD
    return RECORD


def run_challenge(repo_root, model, *, catalog=None):
    catalog = catalog or EvidenceCatalog(repo_root)
    record_id = challenge_record(repo_root)
    snapshot = catalog.snapshot(record_id)
    source = {**snapshot, "id": "E001"}
    prepared, mismatch_check = build_challenge(source)
    before = deepcopy(prepared)
    events = []
    recorded_model = RecordingModel(model)
    engine = ResearchEngine(
        [source], recorded_model, None, events.append,
        max_turns=1, max_seconds=90, max_calls=0,
    )
    reviewed = engine.review(prepared, QUESTION)
    checks = challenge_checks(
        before, reviewed, recorded_model.responses[0],
        engine.model_calls, mismatch_check,
    )
    after = catalog.snapshot(record_id)
    checks["source_snapshot_and_provenance_unchanged"] = snapshot == after
    return {
        "validation_only": True,
        "scope": (
            "One adversarial semantic-review check on authentic saved TORAX "
            "evidence. Injected claims are validation fixtures, not new findings. "
            "No retrieval, simulation, policy change or current-run mutation."
        ),
        "passed": all(checks.values()), "checks": checks,
        "source": source, "record_id": record_id, "question": QUESTION,
        "expected": {"C001": "supported", "C002": "unsupported_or_uncertain",
                     "C003": "unsupported_or_uncertain", "decision": "not_accept"},
        "traceability_validated_input": before,
        "invented_quote_check": mismatch_check,
        "raw_model_response": recorded_model.responses[0],
        "reviewed_report": reviewed, "events": events,
        "usage": engine.total_usage, "model_calls": engine.model_calls,
    }


def main():
    app_root = Path(__file__).resolve().parents[1]
    destination = app_root / ".dashboard/provider-verification/judge-challenge.json"
    started = time.monotonic()
    result = {"started_at": datetime.now(UTC).isoformat(), "validation_only": True}
    try:
        settings = Settings.from_env()
        settings.validate()
        result["settings"] = settings.public()
        result.update(run_challenge(app_root.parent, NebiusModel(settings)))
    except Exception as exc:  # noqa: BLE001 - audit failure without leaking secrets
        result.update(passed=False, error_type=type(exc).__name__)
        if isinstance(exc, ProviderError):
            result["safe_provider_error"] = str(exc)
    result["elapsed_seconds"] = time.monotonic() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"], "checks": result.get("checks", {}),
        "error_type": result.get("error_type"), "report": str(destination),
        "elapsed_seconds": result["elapsed_seconds"],
    }))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
