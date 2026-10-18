"""Read-only research job persistence and the dashboard HTTP boundary."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from dashboard import Dashboard, make_server
from fusion_agent.research import service as service_module
from fusion_agent.research.providers import Settings
from fusion_agent.research.service import ResearchService

RECORD = "results/td3_heating_stability/fixed_partner_comparison.json"
PROTOCOL = "docs/td3_heating_stability.md"
QUESTION = "What should the next bounded heating comparison investigate?"
NEBIUS_SECRET = "test-nebius-secret-not-for-browser"
TAVILY_SECRET = "test-tavily-secret-not-for-browser"


def forbidden_provider(_settings):
    pytest.fail("Offline inspection must not instantiate a network provider")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    record = root / RECORD
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps({"status": "complete", "seeds": [17, 29, 43], "cost": 0.12}),
        encoding="utf-8",
    )
    protocol = root / PROTOCOL
    protocol.parent.mkdir()
    protocol.write_text("Compare local and reset-only controls.", encoding="utf-8")
    # A live training artefact outside the allowlist must stay untouched.
    live = root / "results" / "live-training" / "checkpoint.bin"
    live.parent.mkdir()
    live.write_bytes(b"running-experiment-checkpoint")
    return root


def create_service(repo, state_dir, settings=None, **kwargs):
    return ResearchService(
        repo,
        state_dir,
        settings_factory=lambda: settings or Settings(),
        model_factory=kwargs.get("model_factory", forbidden_provider),
        retrieval_factory=kwargs.get("retrieval_factory", forbidden_provider),
    )


def finish(service):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = service.snapshot()
        if not result["active"]:
            return result
        time.sleep(0.005)
    pytest.fail("The small research fixture did not finish")


def tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_offline_inspection_has_no_provider_calls_or_experiment_writes(repo, tmp_path):
    before = tree_bytes(repo)
    service = create_service(repo, tmp_path / "research")
    started = service.start(QUESTION, [RECORD, PROTOCOL], "offline")
    result = finish(service)
    assert started["run"]["id"] == result["run"]["id"]
    assert result["run"]["status"] == "inspection_complete"
    assert result["run"]["provider"] == {"model": None, "endpoint": None}
    assert result["run"]["report"]["next_runs"] == []
    assert result["run"]["report"]["claims"][0]["evidence"][0]["locator"] == "/status"
    assert any(event["kind"] == "offline" for event in result["run"]["events"])
    assert tree_bytes(repo) == before
    assert all("content" not in item for item in result["run"]["sources"])


def test_source_snapshot_survives_live_file_change_and_is_deep_copied(repo, tmp_path):
    service = create_service(repo, tmp_path / "research")
    service.start(QUESTION, [RECORD], "offline")
    run_id = finish(service)["run"]["id"]
    frozen = service.source(run_id, "E001")
    (repo / RECORD).write_text('{"status":"running","cost":9}', encoding="utf-8")
    assert service.source(run_id, "E001") == frozen
    assert frozen["sha256"] == hashlib.sha256(frozen["content"].encode()).hexdigest()
    altered = service.export(run_id)
    altered["sources"][0]["content"] = "changed by caller"
    assert service.source(run_id, "E001")["content"] == frozen["content"]
    assert service.catalog.snapshot(RECORD)["content"] != frozen["content"]


@pytest.mark.parametrize(
    "ids",
    [[], [RECORD, RECORD], [None], "not-a-list", [f"E{i}" for i in range(9)]],
)
def test_record_selection_is_bounded_and_distinct(repo, tmp_path, ids):
    service = create_service(repo, tmp_path / "research")
    with pytest.raises(ValueError, match="distinct evidence"):
        service.start(QUESTION, ids, "offline")
    assert not service.active
    assert not (service.state_dir / "latest.json").exists()


@pytest.mark.parametrize(
    "record_id",
    [
        "../outside.json",
        "results/td3_heating_stability/../../outside.json",
        "C:/secret.json",
        "results\\td3_heating_stability\\fixed_partner_comparison.json",
        "results/live-training/checkpoint.bin",
        "results/td3_heating_stability/missing/independent_analysis.json",
    ],
)
def test_invalid_and_traversal_records_cannot_start(repo, tmp_path, record_id):
    service = create_service(repo, tmp_path / "research")
    with pytest.raises(ValueError):
        service.start(QUESTION, [record_id], "offline")
    assert not service.active


def test_protocol_only_selection_cannot_masquerade_as_results(repo, tmp_path):
    service = create_service(repo, tmp_path / "research")
    with pytest.raises(ValueError, match="experiment summary"):
        service.start(QUESTION, [PROTOCOL], "offline")


def test_one_active_job_does_not_overwrite_a_running_snapshot(
    repo, tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    original = service_module.offline_report

    def held_report(sources):
        entered.set()
        assert release.wait(5)
        return original(sources)

    monkeypatch.setattr(service_module, "offline_report", held_report)
    service = create_service(repo, tmp_path / "research")
    try:
        run_id = service.start(QUESTION, [RECORD], "offline")["run"]["id"]
        assert entered.wait(3)
        with pytest.raises(ValueError, match="already active"):
            service.start("A different question", [RECORD], "offline")
        assert service.snapshot()["run"]["id"] == run_id
    finally:
        release.set()
        finish(service)
    assert service.snapshot()["run"]["status"] == "inspection_complete"


def test_restarting_marks_an_unfinished_job_interrupted(repo, tmp_path):
    state = tmp_path / "research"
    service = create_service(repo, state)
    service.start(QUESTION, [RECORD], "offline")
    run_id = finish(service)["run"]["id"]
    saved = service.export(run_id)
    saved["status"] = "running"
    (state / run_id / "run.json").write_text(json.dumps(saved), encoding="utf-8")
    restored = create_service(repo, state)
    snapshot = restored.snapshot()
    assert snapshot["run"]["status"] == "interrupted"
    assert not snapshot["active"]
    assert "restarted" in snapshot["run"]["error"]
    assert restored.source(run_id, "E001")["content"] == saved["sources"][0]["content"]
    assert (
        json.loads((state / run_id / "run.json").read_text())["status"] == "interrupted"
    )


@pytest.mark.parametrize("run_id", [None, "../outside", "0" * 31, "G" * 32, "0" * 32])
def test_unknown_export_ids_are_rejected(repo, tmp_path, run_id):
    service = create_service(repo, tmp_path / "research")
    with pytest.raises(ValueError, match="Unknown research run"):
        service.export(run_id)
    with pytest.raises(ValueError, match="Unknown research run"):
        service.source(run_id, "E001")


def test_prior_run_export_and_source_ids_stay_scoped_to_the_saved_run(repo, tmp_path):
    service = create_service(repo, tmp_path / "research")
    service.start(QUESTION, [RECORD], "offline")
    first_id = finish(service)["run"]["id"]
    first_source = service.source(first_id, "E001")
    (repo / RECORD).write_text('{"status":"pending"}', encoding="utf-8")
    service.start("Inspect the new record", [RECORD], "offline")
    second_id = finish(service)["run"]["id"]
    assert first_id != second_id
    assert service.source(first_id, "E001") == first_source
    assert service.source(second_id, "E001") != first_source
    with pytest.raises(ValueError, match="Unknown research source"):
        service.source(first_id, "../run.json")


def test_missing_keys_fail_before_any_job_is_saved(repo, tmp_path):
    service = create_service(repo, tmp_path / "research")
    with pytest.raises(ValueError, match="NEBIUS_API_KEY and TAVILY_API_KEY"):
        service.start(QUESTION, [RECORD], "live")
    assert not service.active
    assert service.snapshot()["run"]["id"] is None


def test_live_provider_failure_redacts_keys_in_saved_and_public_state(
    repo, tmp_path, monkeypatch
):
    settings = Settings(nebius_key=NEBIUS_SECRET, tavily_key=TAVILY_SECRET)

    class FailedEngine:
        def __init__(self, *_args):
            pass

        def run(self, _question):
            raise RuntimeError(f"Provider failed: {NEBIUS_SECRET} {TAVILY_SECRET}")

    monkeypatch.setattr(service_module, "ResearchEngine", FailedEngine)
    service = create_service(
        repo,
        tmp_path / "research",
        settings,
        model_factory=lambda _: object(),
        retrieval_factory=lambda _: object(),
    )
    service.start(QUESTION, [RECORD], "live")
    public = finish(service)
    assert public["run"]["status"] == "failed"
    assert "[redacted]" in public["run"]["error"]
    persisted = service.export(public["run"]["id"])
    serialized = json.dumps([public, persisted])
    assert NEBIUS_SECRET not in serialized
    assert TAVILY_SECRET not in serialized


def test_statusless_summary_still_completes_an_explicit_offline_inspection(
    repo, tmp_path
):
    (repo / RECORD).write_text('{"cost":0.12,"seeds":[17,29]}', encoding="utf-8")
    service = create_service(repo, tmp_path / "research")
    service.start(QUESTION, [RECORD], "offline")
    result = finish(service)
    assert result["run"]["status"] == "inspection_complete"
    assert result["run"]["report"]["next_runs"] == []


def test_late_parallel_agent_events_cannot_contaminate_a_new_run(
    repo, tmp_path, monkeypatch
):
    callbacks = []

    class FailedParallelEngine:
        def __init__(self, sources, model, retrieval, emit):
            callbacks.append(emit)

        def run(self, question):
            raise RuntimeError("One researcher failed while another was pending")

    monkeypatch.setattr(service_module, "ResearchEngine", FailedParallelEngine)
    service = create_service(
        repo,
        tmp_path / "research",
        Settings(nebius_key=NEBIUS_SECRET, tavily_key=TAVILY_SECRET),
        model_factory=lambda _: object(),
        retrieval_factory=lambda _: object(),
    )
    service.start(QUESTION, [RECORD], "live")
    first_id = finish(service)["run"]["id"]
    failed = service.export(first_id)
    callbacks[0]({"kind": "late", "message": "Old provider response"})
    assert service.export(first_id) == failed

    service.start("Inspect again", [RECORD], "offline")
    second_id = finish(service)["run"]["id"]
    completed = service.export(second_id)
    callbacks[0]({"kind": "late", "message": "Old source must not enter this run"})
    assert service.export(second_id) == completed
    assert service.export(first_id) == failed


@pytest.fixture
def http_app(repo, tmp_path):
    app = Dashboard(tmp_path / "dashboard")
    settings = Settings(nebius_key=NEBIUS_SECRET, tavily_key=TAVILY_SECRET)
    app.research = create_service(repo, tmp_path / "http-research", settings)
    server = make_server(app, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield app, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def post(base, body, headers=None):
    base_headers = {
        "Content-Type": "application/json",
        "X-Fusion-UI": "1",
        "Origin": base,
    }
    base_headers.update(headers or {})
    base_headers = {
        key: value for key, value in base_headers.items() if value is not None
    }
    return urlopen(
        Request(
            base + "/api/research",
            method="POST",
            data=json.dumps(body).encode(),
            headers=base_headers,
        ),
        timeout=3,
    )


def test_research_http_offline_flow_assets_export_and_key_redaction(http_app):
    app, base = http_app
    with urlopen(base + "/api/research", timeout=3) as response:
        public = response.read().decode()
        assert json.loads(public)["configured"]["nebius"] is True
        assert NEBIUS_SECRET not in public and TAVILY_SECRET not in public
    for path, mime in [
        ("/research.js", "text/javascript"),
        ("/research.css", "text/css"),
    ]:
        with urlopen(base + path, timeout=3) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith(mime)
            assert len(response.read()) > 100
    with post(
        base, {"question": QUESTION, "record_ids": [RECORD], "mode": "offline"}
    ) as response:
        assert response.status == 202
    run_id = finish(app.research)["run"]["id"]
    for path, query in [
        ("/api/research/export", {"run_id": run_id}),
        ("/api/research/source", {"run_id": run_id, "source_id": "E001"}),
    ]:
        with urlopen(base + path + "?" + urlencode(query), timeout=3) as response:
            output = response.read().decode()
            assert NEBIUS_SECRET not in output and TAVILY_SECRET not in output
            assert "complete" in output
    assert not app.active  # Research never starts the pre-existing Flower run.


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://attacker.example"},
        {"Origin": None},
        {"X-Fusion-UI": None},
        {"Content-Type": "text/plain"},
        {"Host": "attacker.example"},
    ],
)
def test_research_post_enforces_same_origin_guards(http_app, headers):
    app, base = http_app
    request_headers = {
        "Content-Type": "application/json",
        "X-Fusion-UI": "1",
        "Origin": base,
    }
    request_headers.update(headers)
    request_headers = {
        key: value for key, value in request_headers.items() if value is not None
    }
    with pytest.raises(HTTPError) as error:
        # No unread body: Windows may reset a socket closed on rejected headers
        # before urllib can receive its 403 response when a body is in flight.
        urlopen(
            Request(base + "/api/research", method="POST", headers=request_headers),
            timeout=3,
        )
    assert error.value.code == 403
    assert app.research.snapshot()["run"]["id"] is None


@pytest.mark.parametrize(
    "path",
    [
        "/api/research/export",
        "/api/research/export?run_id=..%2Foutside",
        "/api/research/source?run_id=" + "0" * 32 + "&source_id=E001",
        "/api/research/export?run_id=" + "0" * 32,
    ],
)
def test_http_sources_and_exports_require_a_known_run(http_app, path):
    _, base = http_app
    with pytest.raises(HTTPError) as error:
        urlopen(base + path, timeout=3)
    assert error.value.code == 404


def test_http_missing_keys_error_does_not_launch_or_expose_secrets(http_app):
    app, base = http_app
    app.research.settings_factory = Settings
    with pytest.raises(HTTPError) as error:
        post(base, {"question": QUESTION, "record_ids": [RECORD], "mode": "live"})
    assert error.value.code == 400
    payload = error.value.read().decode()
    assert "NEBIUS_API_KEY" in payload
    assert NEBIUS_SECRET not in payload and TAVILY_SECRET not in payload
    assert not app.research.active


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"question": "", "record_ids": [RECORD]},
        {"question": QUESTION, "record_ids": [RECORD], "mode": "execute"},
    ],
)
def test_http_invalid_research_requests_return_errors(http_app, body):
    app, base = http_app
    with pytest.raises(HTTPError) as error:
        post(base, body)
    assert error.value.code == 400
    assert not app.research.active
