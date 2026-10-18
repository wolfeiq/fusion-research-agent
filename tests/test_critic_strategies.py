"""Control equivalence, protocol validation and clipping without target data."""
import copy
import json

import numpy as np
import pytest

from hfmarl.federation.critic_strategies import calibrate_thresholds, aggregate_variants
from hfmarl.federation.representation import RepresentationDelta, model_hash


def setup_round(values=None, dtype=np.float32):
    values = values if values is not None else [(2., 4.), (4., 2.)]
    base = {"q0.weight": np.array([10., 20.], dtype=dtype),
            "q0.bias": np.array([30.], dtype=dtype),
            "q1.weight": np.array([5.], dtype=dtype)}
    cohort = [f"source_{index}" for index in range(len(values))]
    revisions = {source: "config-v1" for source in cohort}
    updates = [RepresentationDelta(f"calibration/{source}", source, model_hash(base),
                                  "schema", revisions[source],
                                  {name: np.full_like(value, q0 if name.startswith("q0.") else q1)
                                   for name, value in base.items()})
               for source, (q0, q1) in zip(cohort, values)]
    return base, updates, cohort, "schema", revisions


def test_calibration_is_branchwise_source_median_with_provenance():
    args = setup_round()
    result = calibrate_thresholds(*args)
    assert result["thresholds"] == pytest.approx({"q0": 3 * np.sqrt(3), "q1": 3.})
    metadata = result["diagnostics"]
    assert metadata["target_data_used"] is False
    assert metadata["base_hash"] == model_hash(args[0])
    assert metadata["source_norms"]["source_0"]["q0"] == pytest.approx(2 * np.sqrt(3))
    assert metadata["source_ids"] == args[2]
    assert metadata["update_ids"] == [update.update_id for update in args[1]]
    json.dumps(result, allow_nan=False)


def test_uniform_clipped_and_norm_matched_have_expected_values():
    args = setup_round()
    result = aggregate_variants(*args, {"q0": 2 * np.sqrt(3), "q1": 2.})
    proposals, meta = result["proposals"], result["diagnostics"]
    for name, base in args[0].items():
        np.testing.assert_allclose(proposals["uniform"][name], base + 3)
        np.testing.assert_allclose(proposals["clipped"][name], base + 2)
        np.testing.assert_allclose(proposals["norm_matched_uniform"][name], base + 2)
        assert proposals["uniform"][name].dtype == base.dtype
    assert meta["clipping_factors"]["source_1"]["q0"] == pytest.approx(.5)
    assert meta["clipping_factors"]["source_0"]["q1"] == pytest.approx(.5)
    assert meta["norm_match_factors"] == pytest.approx({"q0": 2 / 3, "q1": 2 / 3})
    assert all(meta["norm_match_available"].values())
    assert not any(meta["norm_match_amplified"].values())
    json.dumps(meta, allow_nan=False)


def test_no_clipping_is_an_exact_equal_weight_control():
    args = setup_round()
    result = aggregate_variants(*args, {"q0": 100., "q1": 100.})
    for name, value in result["proposals"]["uniform"].items():
        np.testing.assert_array_equal(value, result["proposals"]["clipped"][name])
        np.testing.assert_array_equal(value, result["proposals"]["norm_matched_uniform"][name])


def test_zero_deltas_and_zero_threshold_produce_base_without_division():
    args = setup_round([(0., 0.), (0., 0.)])
    calibration = calibrate_thresholds(*args)
    assert calibration["thresholds"] == {"q0": 0., "q1": 0.}
    result = aggregate_variants(*args, calibration["thresholds"])
    for proposal in result["proposals"].values():
        for name, value in args[0].items():
            np.testing.assert_array_equal(proposal[name], value)
    assert result["diagnostics"]["norm_match_factors"] == {"q0": 0., "q1": 0.}
    assert all(result["diagnostics"]["norm_match_available"].values())
    nonzero = setup_round()
    clipped = aggregate_variants(*nonzero, {"q0": 0., "q1": 0.})
    for name, value in nonzero[0].items():
        np.testing.assert_array_equal(clipped["proposals"]["clipped"][name], value)


def test_cancellation_can_require_amplification_or_make_norm_match_impossible():
    args = setup_round([(1., 10.), (1., -6.), (-2., -3.)], dtype=np.float64)
    result = aggregate_variants(*args, {"q0": np.sqrt(3), "q1": 3.})
    meta = result["diagnostics"]
    assert meta["norm_match_available"] == {"q0": False, "q1": True}
    assert meta["norm_match_factors"]["q0"] is None
    assert meta["norm_match_factors"]["q1"] == pytest.approx(3.)
    assert meta["norm_match_amplified"]["q1"] is True
    for name, base in args[0].items():
        matched = result["proposals"]["norm_matched_uniform"][name]
        np.testing.assert_allclose(matched, base if name.startswith("q0.") else base + 1.)
    # Clipping can reverse the mean direction; norm matching preserves it.
    np.testing.assert_allclose(result["proposals"]["clipped"]["q1.weight"], args[0]["q1.weight"] - 1.)
    json.dumps(meta, allow_nan=False)


def test_inputs_are_immutable_and_source_order_uses_registered_cohort():
    args = setup_round()
    before = copy.deepcopy(args)
    thresholds = {"q0": 1., "q1": 2.}
    first = aggregate_variants(*args, thresholds)
    shuffled = aggregate_variants(args[0], list(reversed(args[1])), *args[2:], thresholds)
    assert first["diagnostics"] == shuffled["diagnostics"]
    for proposal in first["proposals"].values():
        for name, value in proposal.items():
            assert not np.shares_memory(value, args[0][name])
            assert all(not np.shares_memory(value, update.tensors[name]) for update in args[1])
    for name, value in args[0].items():
        np.testing.assert_array_equal(value, before[0][name])
    for actual, original in zip(args[1], before[1]):
        assert actual.update_id == original.update_id and actual.base_hash == original.base_hash
        for name, value in actual.tensors.items():
            np.testing.assert_array_equal(value, original.tensors[name])
    assert thresholds == {"q0": 1., "q1": 2.}


@pytest.mark.parametrize("bad", ["base_hash", "schema", "revision", "shape", "dtype",
                                 "nan", "infinity", "duplicate_id", "duplicate_source",
                                 "missing_source", "missing_branch", "unknown_source"])
def test_invalid_envelopes_rejected_by_both_apis(bad):
    args = setup_round()
    base, updates, cohort, schema, revisions = args
    if bad == "base_hash": updates[0].base_hash = "wrong"
    if bad == "schema": updates[0].schema = "wrong"
    if bad == "revision": updates[0].revision = "wrong"
    if bad == "shape": updates[0].tensors["q0.weight"] = np.zeros(7, np.float32)
    if bad == "dtype": updates[0].tensors["q0.weight"] = updates[0].tensors["q0.weight"].astype(np.float64)
    if bad == "nan": updates[0].tensors["q0.weight"][0] = np.nan
    if bad == "infinity": updates[0].tensors["q0.weight"][0] = np.inf
    if bad == "duplicate_id": updates[1].update_id = updates[0].update_id
    if bad == "duplicate_source": updates[1].source = updates[0].source
    if bad == "missing_source": updates.pop()
    if bad == "missing_branch": updates[0].tensors.pop("q1.weight")
    if bad == "unknown_source": updates[0].source = "unknown"
    with pytest.raises(ValueError):
        calibrate_thresholds(*args)
    with pytest.raises(ValueError):
        aggregate_variants(*args, {"q0": 1., "q1": 1.})


@pytest.mark.parametrize("thresholds", [{"q0": 1.}, {"q0": 1., "q1": 1., "q2": 1.},
                                        {"q0": -1., "q1": 1.}, {"q0": np.nan, "q1": 1.},
                                        {"q0": np.inf, "q1": 1.}, {"q0": True, "q1": 1.},
                                        {"q0": "1", "q1": 1.}])
def test_invalid_frozen_thresholds_rejected(thresholds):
    with pytest.raises(ValueError):
        aggregate_variants(*setup_round(), thresholds)


def test_large_float32_delta_norm_is_computed_in_float64():
    args = setup_round([(1e20, 1e20), (1e20, 1e20)])
    calibration = calibrate_thresholds(*args)
    assert calibration["thresholds"]["q0"] == pytest.approx(np.sqrt(3) * 1e20)
    assert np.isfinite(calibration["thresholds"]["q0"])


@pytest.mark.parametrize("bad", ["single_branch", "unknown_branch", "integer", "empty", "nan"])
def test_invalid_base_contract_rejected(bad):
    base, updates, cohort, schema, revisions = setup_round()
    if bad == "single_branch": base.pop("q1.weight")
    if bad == "unknown_branch": base["actor.weight"] = np.zeros(1, np.float32)
    if bad == "integer": base["q0.weight"] = np.zeros(2, np.int32)
    if bad == "empty": base["q0.weight"] = np.zeros(0, np.float32)
    if bad == "nan": base["q0.weight"][0] = np.nan
    with pytest.raises(ValueError):
        calibrate_thresholds(base, updates, cohort, schema, revisions)


def test_large_representable_mean_passes_protocol_validation():
    args = setup_round([(3e38, 3e38), (3e38, 3e38)])
    with np.errstate(over="raise", invalid="raise"):
        result = aggregate_variants(*args, {"q0": 1., "q1": 1.})
    for name, value in result["proposals"]["uniform"].items():
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, np.full_like(value, 3e38))


def test_unrepresentable_base_plus_mean_rejected():
    args = setup_round([(3e38, 3e38), (3e38, 3e38)])
    for value in args[0].values():
        value.fill(3e38)
    for update in args[1]:
        update.base_hash = model_hash(args[0])
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="Nonfinite aggregate"):
            aggregate_variants(*args, {"q0": 1., "q1": 1.})
