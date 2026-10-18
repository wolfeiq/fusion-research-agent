"""One judge-only regression against unchanged captured run13 evidence.

Requires --run-json; reads no credential files and performs no retrieval or RL.
Provider environment is supplied by the caller. The output retains only cited
claim fixtures and review diagnostics, never whole web or experiment snapshots.
"""

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.providers import NebiusModel, ProviderError, Settings
from fusion_agent.research.verification import validate_report

RUN_ID = "42bdaaf230724cb4ae5eea94c54cfd78"
RUN_SHA256 = "e8c5567c98e8a2c3cc20e0aa0e73f66685dcff092fcf4cb053a9470e2992ee3b"
SOURCE_HASHES = {
    "E001": "cfbf759cdea183d9c8fb8ac8788e3204a6e3389ffd91ca0e5ee097d727c09431",
    "W004": "db4092d68f9c936216c60c48eff2ce27a45894162af8baa196510bd98fa12594",
}
QUESTION = (
    "Validation-only citation-scope review, not a new research conclusion. "
    "Judge each supplied claim independently against its own exact cited passage "
    "or JSON locator in context. Sources and original quotations are unchanged. "
    "Do not assume any claim is correct; return the ordinary review_claims verdicts "
    "and overall decision. No retrieval, experiment, or proposal execution is needed."
)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_saved_run(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8_000_000 or sha256(raw) != RUN_SHA256:
        raise ValueError("Run bytes do not match the frozen citation-scope regression")
    run = json.loads(raw)
    if run.get("id") != RUN_ID or run.get("status") != "completed":
        raise ValueError("Expected the completed frozen run")
    sources = {source["id"]: source for source in run["sources"]}
    selected = []
    for identifier, expected in SOURCE_HASHES.items():
        source = sources[identifier]
        if source["sha256"] != expected or sha256(source["content"].encode()) != expected:
            raise ValueError("Captured source content hash changed")
        selected.append(deepcopy(source))
    return run, selected, sha256(raw)


def build_challenge(run, sources):
    """Keep original defective claims; add one independently scoped control."""
    originals = {claim["id"]: claim for claim in run["report"]["claims"]}
    keys = ("id", "text", "kind", "confidence", "evidence")
    claims = []
    for identifier in ("C1", "C2", "C4"):
        claim = {key: deepcopy(originals[identifier][key]) for key in keys}
        claim["evidence"] = [{key: citation[key] for key in ("source_id", "quote", "locator")}
                             for citation in claim["evidence"]]
        claims.append(claim)
    if claims[0]["evidence"][0]["locator"] != "/round_aggregation_diagnostics/records/3":
        raise ValueError("Original candidate citation changed")
    control = {
        "id": "CONTROL", "kind": "observation", "confidence": "high",
        "text": "In candidate seed 17, round 1, the clipped deposition_sac record reports donor_cool and donor_warm clipped in both q0 and q1, while cold_target has clipping factor 1.0 in both critics.",
        "evidence": deepcopy(claims[0]["evidence"]),
    }
    prepared = validate_report({
        "answer": "Validation-only citation-scope fixtures.",
        "claims": [control, *claims], "next_runs": [],
        "limitations": ["One semantic-review regression cannot establish general judge reliability."],
        "stop_reason": "One bounded judge review, no new research or policy activation.",
    }, sources)
    if any(claim["verification"]["status"] != "traceable" for claim in prepared["claims"]):
        raise ValueError("All fixtures must first pass exact citation traceability")
    return prepared


class OneCallModel:
    def __init__(self, model):
        self.model = model
        self.responses = []

    def complete(self, messages, tools, tool_choice="auto"):
        if self.responses:
            raise RuntimeError("Only one judge call is authorized by this probe")
        message, usage = self.model.complete(messages, tools, tool_choice)
        self.responses.append(deepcopy(message))
        return message, usage


def raw_review(message):
    try:
        calls = message["tool_calls"]
        if len(calls) != 1 or calls[0]["function"]["name"] != "review_claims":
            raise ValueError("Expected one review call")
        value = json.loads(calls[0]["function"]["arguments"])
        return value if isinstance(value, dict) else {}
    except (KeyError, IndexError, ValueError, TypeError):
        return {}


def run_challenge(run_json, model):
    run, sources, before_hash = load_saved_run(run_json)
    prepared = build_challenge(run, sources)
    before = deepcopy(prepared)
    source_before = deepcopy(sources)
    recorded = OneCallModel(model)
    engine = ResearchEngine(sources, recorded, None, lambda _event: None,
                            max_turns=1, max_seconds=90, max_calls=0)
    reviewed = engine.review(prepared, QUESTION)
    original_response = recorded.responses[0]
    raw = raw_review(original_response)
    lookup = {claim["id"]: claim for claim in reviewed["claims"]}
    checks = {
        "all_fixtures_initially_traceable": all(claim["verification"]["status"] == "traceable" for claim in before["claims"]),
        "narrow_control_supported_and_retained": (
            lookup["CONTROL"].get("review", {}).get("verdict") == "supported"
            and lookup["CONTROL"]["verification"]["status"] == "traceable"
            and lookup["CONTROL"]["text"] in reviewed["answer"]),
        **{f"{identifier}_coverage_or_scope_rejected": (
            lookup[identifier].get("review", {}).get("verdict") in {"uncertain", "unsupported"}
            and lookup[identifier]["verification"]["status"] == "unsupported"
            and lookup[identifier]["text"] not in reviewed["answer"])
           for identifier in ("C1", "C2", "C4")},
        "raw_model_decision_not_accept": raw.get("decision") in {"revise", "insufficient"},
        "guarded_decision_not_accept": reviewed["judge"]["decision"] in {"revise", "insufficient"},
        "exactly_one_judge_call": engine.model_calls == len(recorded.responses) == 1,
        "captured_sources_unchanged": sources == source_before,
        "run_bytes_unchanged": sha256(Path(run_json).read_bytes()) == before_hash,
    }
    return {
        "schema": "fusion-citation-scope-probe-v1", "validation_only": True,
        "scope": "One model review using authentic unchanged captured sources; no web retrieval, physics run, policy change, or historical report rewrite. Fixture claims are not new findings.",
        "passed": all(checks.values()), "checks": checks, "run_id": run["id"],
        "run_sha256": before_hash,
        "sources": [{"id": source["id"], "kind": source["kind"], "sha256": source["sha256"],
                     "uri": source["uri"] if source["kind"] == "web" else "captured-local:E001",
                     "captured_characters": len(source["content"])} for source in sources],
        "traceability_validated_input": before, "reviewed_report": reviewed,
        "raw_review": raw, "raw_response_sha256": sha256(json.dumps(original_response, sort_keys=True).encode()),
        "expected": {"CONTROL": "supported", "C1": "uncertain_or_unsupported",
                     "C2": "uncertain_or_unsupported", "C4": "uncertain_or_unsupported",
                     "decision": "not_accept"},
        "model_calls": engine.model_calls, "usage": engine.total_usage,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    result = {"validation_only": True, "passed": False}
    try:
        settings = Settings.from_env()
        settings.validate()
        result["settings"] = settings.public()
        result.update(run_challenge(args.run_json, NebiusModel(settings)))
    except Exception as exc:  # noqa: BLE001 - do not expose credentials or provider bodies
        result["error_type"] = type(exc).__name__
        if isinstance(exc, ProviderError):
            result["safe_provider_error"] = str(exc)
    result["elapsed_seconds"] = time.monotonic() - started
    output = Path(__file__).resolve().parents[1] / ".dashboard/provider-verification/judge-citation-scope.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result.get("checks", {}),
                      "error_type": result.get("error_type"), "report": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
