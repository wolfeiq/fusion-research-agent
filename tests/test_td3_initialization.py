"""Calibrated initial power is explicit and does not perturb legacy randomness."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfmarl.agents.thermal_td3 import TD3, TD3Config


def learner(fractions=None, seed=17):
    cfg = TD3Config(hidden=8, batch=4, actor_init_fraction=fractions)
    return TD3(9, np.zeros(2, np.float32), np.array([.25, .05], np.float32), seed, cfg)


@pytest.mark.parametrize("fractions", [(.5, .5), (.2, .8), (.1, .7)])
def test_explicit_initial_fraction_constant_for_every_observation_and_target(fractions):
    model = learner(fractions)
    observations = torch.tensor(np.random.default_rng(18).normal(size=(40, 9)), dtype=torch.float32)
    expected = np.broadcast_to(2 * np.array(fractions) - 1, (40, 2))
    actual = model.actor(observations).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=6e-8, rtol=0)
    assert torch.equal(model.actor(observations), model.target_actor(observations))
    assert torch.count_nonzero(model.actor[-2].weight) == 0
    if fractions == (.5, .5):
        np.testing.assert_array_equal(actual, np.zeros((40, 2), np.float32))


@pytest.mark.parametrize("fractions", [(), (.5,), (.5, .5, .5), [.5, .5],
                                       (0., .5), (1., .5), (-.1, .5), (1.1, .5),
                                       (np.nan, .5), (np.inf, .5), (True, .5),
                                       (np.bool_(False), .5), (".5", .5)])
def test_invalid_initial_fractions_fail_before_constructing_a_learner(fractions):
    with pytest.raises(ValueError, match="actor_init_fraction"):
        TD3Config(actor_init_fraction=fractions)


def test_explicit_initialization_preserves_hidden_layers_critics_and_rng_streams():
    state = torch.random.get_rng_state().clone()
    legacy = learner()
    calibrated = learner((.5, .5))
    assert torch.equal(state, torch.random.get_rng_state())
    for key, value in legacy.actor.state_dict().items():
        if key not in {"4.weight", "4.bias"}:
            assert torch.equal(value, calibrated.actor.state_dict()[key])
    for name in ("critics", "target_critics"):
        for key, value in getattr(legacy, name).state_dict().items():
            assert torch.equal(value, getattr(calibrated, name).state_dict()[key])
    for name in ("rng", "replay_rng"):
        np.testing.assert_array_equal(getattr(legacy, name).integers(10000, size=20),
                                      getattr(calibrated, name).integers(10000, size=20))
    assert torch.equal(legacy.torch_rng.get_state(), calibrated.torch_rng.get_state())


def test_none_retains_observation_dependent_legacy_initialization():
    model = learner()
    assert torch.count_nonzero(model.actor[-2].weight) > 0
    assert torch.equal(model.actor[-2].bias, torch.tensor([-1., -1.]))
    assert torch.equal(model.actor[-2].weight, model.target_actor[-2].weight)


def test_seeded_updates_remain_reproducible_after_calibrated_initialization():
    first, second = learner((.5, .5)), learner((.5, .5))
    rng = np.random.default_rng(912)
    for index in range(10):
        row = (rng.normal(size=9).astype(np.float32), rng.uniform(-1, 1, 2).astype(np.float32),
               -float(index), rng.normal(size=9).astype(np.float32), index == 9)
        for model in (first, second):
            model.remember(*row)
    for _ in range(8):
        first.train_step()
        second.train_step()
    for name in ("actor", "target_actor", "critics", "target_critics"):
        for key, value in getattr(first, name).state_dict().items():
            assert torch.equal(value, getattr(second, name).state_dict()[key])
