"""Pinned Flower 1.22 NumPyClient transport for role-separated RL deltas.

Uses Flower's legacy gRPC execution API, not SuperGrid. A local experiment runner
must restrict insecure listeners to loopback. Client callbacks own durable local
learner state and idempotency; only critic deltas leave the client.
"""
import json

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from .repeated import ROLES, TensorWire
from .representation import RepresentationDelta


class RoleClient(fl.client.NumPyClient):
    """Backend methods: contribute(bases, round), accept(proposal, round).

    Backends must cache completed round results atomically with local learner
    state, so retried Flower messages do not repeat training or installation.
    """
    def __init__(self, backend, bases):
        self.backend = backend
        self.wire = TensorWire(bases)

    def get_parameters(self, config):
        raise RuntimeError("Coordinator must provide the registered initial bases")

    def fit(self, parameters, config):
        bases = self.wire.decode(parameters)
        round_id = int(config["round"])
        if config["recover"]:
            return parameters, 1, {"source": self.backend.source, "recovery": True}
        envelopes = self.backend.contribute(bases, round_id)
        metadata = {}
        for role, update in envelopes.items():
            metadata[role] = {key: getattr(update, key) for key in
                              ("update_id", "source", "base_hash", "schema", "revision")}
        return self.wire.encode({r: envelopes[r].tensors for r in ROLES}), 1, {
            "source": self.backend.source, "envelopes": json.dumps(metadata),
        }

    def evaluate(self, parameters, config):
        report = self.backend.accept(self.wire.decode(parameters), int(config["round"]))
        # Loss is an audit summary; it never weights aggregation.
        return float(report["deployed_cost"]), 1, {
            "source": self.backend.source,
            "report": json.dumps(report, allow_nan=False, sort_keys=True),
        }


class RoleStrategy(fl.server.strategy.FedAvg):
    def __init__(self, coordinator, checkpoint):
        self.coordinator, self.checkpoint = coordinator, checkpoint
        self.wire = TensorWire(coordinator.bases())
        count = len(coordinator.channels[ROLES[0]].cohort)
        self.recovering = coordinator.pending
        super().__init__(
            fraction_fit=1., fraction_evaluate=1., min_fit_clients=count,
            min_evaluate_clients=count, min_available_clients=count,
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(self.wire.encode(coordinator.bases())),
            on_fit_config_fn=self.fit_config, on_evaluate_config_fn=self.eval_config,
        )

    def fit_config(self, _):
        self.recovering = self.coordinator.pending
        return {"round": self.coordinator.version + (not self.recovering),
                "recover": self.recovering}

    def eval_config(self, _):
        return {"round": self.coordinator.version}

    def _validate_results(self, results, failures):
        cohort = self.coordinator.channels[ROLES[0]].cohort
        sources = [result.metrics.get("source") for _, result in results]
        if failures or len(sources) != len(cohort) or set(sources) != set(cohort):
            raise RuntimeError("Failed, duplicate, or incomplete registered cohort")

    def aggregate_fit(self, server_round, results, failures):
        self._validate_results(results, failures)
        if not self.recovering:
            envelopes = {r: [] for r in ROLES}
            for _, result in results:
                values = self.wire.decode(parameters_to_ndarrays(result.parameters))
                meta = json.loads(result.metrics["envelopes"])
                if set(meta) != set(ROLES):
                    raise ValueError("Both channel identities are required")
                for role in ROLES:
                    if meta[role]["source"] != result.metrics["source"]:
                        raise ValueError("Source identity mismatch")
                    envelopes[role].append(RepresentationDelta(**meta[role], tensors=values[role]))
            self.coordinator.aggregate(envelopes)
            self.coordinator.save(self.checkpoint)
        elif not all(result.metrics.get("recovery") for _, result in results):
            raise ValueError("Recovery cannot train another round")
        return ndarrays_to_parameters(self.wire.encode(self.coordinator.bases())), {}

    def aggregate_evaluate(self, server_round, results, failures):
        self._validate_results(results, failures)
        metrics = {res.metrics["source"]: json.loads(res.metrics["report"])
                   for _, res in results}
        self.coordinator.acknowledge(metrics)
        self.coordinator.save(self.checkpoint)
        return sum(float(r.loss) for _, r in results) / len(results), {}
