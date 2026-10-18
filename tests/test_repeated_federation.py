import copy

import numpy as np
import pytest

from hfmarl.federation.repeated import ROLES, RepeatedCoordinator, TensorWire
from hfmarl.federation.representation import RepresentationDelta, model_hash


def coordinator(strategy="uniform"):
    bases = {r: {"q0.weight": np.zeros(2, np.float32),
                 "q1.weight": np.zeros(2, np.float32)} for r in ROLES}
    return RepeatedCoordinator(
        bases, ["a", "b"], {r: r + "-schema" for r in ROLES},
        {"a": "v1", "b": "v1"}, strategy=strategy,
        thresholds={r: {"q0": 1., "q1": 1.} for r in ROLES}, run_id="test",
    )


def updates(server):
    return {role: [RepresentationDelta(
        f"test:{server.version + 1}:{source}:{role}", source,
        model_hash(channel.tensors), channel.schema, "v1",
        {k: np.full_like(v, i + 1) for k, v in channel.tensors.items()},
    ) for i, source in enumerate(channel.cohort)] for role, channel in server.channels.items()}


def test_two_rounds_resume_and_consume_once(tmp_path):
    server = coordinator()
    old = updates(server)
    server.aggregate(old)
    server.save(tmp_path / "server.npz")
    restored = RepeatedCoordinator.load(tmp_path / "server.npz")
    assert restored.version == 1 and restored.pending
    with pytest.raises(RuntimeError):
        restored.aggregate(old)
    restored.acknowledge({"a": {"accepted": False}, "b": {"accepted": True}})
    with pytest.raises(ValueError):
        restored.aggregate(old)
    restored.aggregate(updates(restored))
    for channel in restored.channels.values():
        assert len(channel.consumed) == 4
        for tensor in channel.tensors.values():
            np.testing.assert_array_equal(tensor, [3., 3.])


@pytest.mark.parametrize("failure", ["schema", "nonfinite", "missing", "source", "base"])
def test_second_role_failure_does_not_apply_first(failure):
    server = coordinator()
    envelopes = updates(server)
    bad = envelopes[ROLES[1]][0]
    if failure == "schema":
        bad.schema = envelopes[ROLES[0]][0].schema
    elif failure == "nonfinite":
        bad.tensors["q0.weight"][0] = np.nan
    elif failure == "missing":
        envelopes[ROLES[1]].pop()
    elif failure == "source":
        bad.source = "unknown"
    else:
        bad.base_hash = "wrong"
    with pytest.raises(ValueError):
        server.aggregate(envelopes)
    assert server.version == 0 and not server.pending and not server.history
    assert all(not c.consumed for c in server.channels.values())
    assert all(np.all(v == 0) for c in server.channels.values() for v in c.tensors.values())


def test_update_ids_cannot_be_swapped_between_registered_sources():
    server = coordinator()
    envelopes = updates(server)
    first, second = envelopes[ROLES[1]]
    first.update_id, second.update_id = second.update_id, first.update_id
    with pytest.raises(ValueError, match="own source"):
        server.aggregate(envelopes)
    assert server.version == 0 and not server.pending
    assert all(not channel.consumed for channel in server.channels.values())


def test_clipping_changes_only_delta_norm_not_branch_or_client_weight():
    server = coordinator("clipped")
    server.aggregate(updates(server))
    for channel in server.channels.values():
        for tensor in channel.tensors.values():
            assert np.linalg.norm(tensor) == pytest.approx(1.)
    assert server.history[0]["channels"][ROLES[0]]["clipping_factors"]["b"]["q0"] < 1


def test_ack_requires_full_finite_cohort():
    server = coordinator()
    server.aggregate(updates(server))
    for report in ({"a": {}}, {"a": {"cost": np.nan}, "b": {}}):
        with pytest.raises(ValueError):
            server.acknowledge(report)
        assert server.pending


def test_wire_rejects_extra_parameters_and_schema_shape():
    values = coordinator().bases()
    wire = TensorWire(values)
    arrays = wire.encode(values)
    round_trip = wire.decode(arrays)
    arrays[0][0] = 10
    assert round_trip[ROLES[0]]["q0.weight"][0] == 0
    extra = copy.deepcopy(values)
    extra[ROLES[0]]["actor.weight"] = np.ones(2, np.float32)
    with pytest.raises(ValueError):
        wire.encode(extra)
    with pytest.raises(ValueError):
        wire.decode(arrays + [arrays[0]])


def test_flower_strategy_serialization_and_recovery(tmp_path):
    fl = pytest.importorskip("flwr")
    from hfmarl.federation.flower_runtime import RoleClient, RoleStrategy

    class Backend:
        def __init__(self, source, server):
            self.source, self.server, self.calls = source, server, 0

        def contribute(self, bases, round_id):
            self.calls += 1
            return {r: next(u for u in us if u.source == self.source)
                    for r, us in updates(self.server).items()}

        def accept(self, proposal, round_id):
            return {"deployed_cost": 1., "accepted": False, "round": round_id}

    server = coordinator()
    strategy = RoleStrategy(server, tmp_path / "coordinator.npz")
    clients = [RoleClient(Backend(s, server), server.bases()) for s in ("a", "b")]
    parameters = fl.common.ndarrays_to_parameters(strategy.wire.encode(server.bases()))
    status = fl.common.Status(code=fl.common.Code.OK, message="")
    cfg = strategy.fit_config(1)
    replies = []
    for client in clients:
        arrays, n, meta = client.fit(fl.common.parameters_to_ndarrays(parameters), cfg)
        replies.append((None, fl.common.FitRes(status,
            fl.common.ndarrays_to_parameters(arrays), n, meta)))
    proposal, _ = strategy.aggregate_fit(1, replies, [])
    assert server.pending
    resumed = RoleStrategy(RepeatedCoordinator.load(tmp_path / "coordinator.npz"),
                           tmp_path / "coordinator.npz")
    cfg = resumed.fit_config(1)
    assert cfg == {"round": 1, "recover": True}
    recovery = []
    for client in clients:
        arrays, n, meta = client.fit(fl.common.parameters_to_ndarrays(proposal), cfg)
        assert client.backend.calls == 1
        recovery.append((None, fl.common.FitRes(status,
            fl.common.ndarrays_to_parameters(arrays), n, meta)))
    proposal, _ = resumed.aggregate_fit(1, recovery, [])
    evaluations = []
    for client in clients:
        cost, n, meta = client.evaluate(fl.common.parameters_to_ndarrays(proposal),
                                        resumed.eval_config(1))
        evaluations.append((None, fl.common.EvaluateRes(status, cost, n, meta)))
    resumed.aggregate_evaluate(1, evaluations, [])
    assert resumed.coordinator.version == 1 and not resumed.coordinator.pending
