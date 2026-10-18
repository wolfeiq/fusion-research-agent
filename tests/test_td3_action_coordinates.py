"""Explicit action-coordinate and actor burn-in candidates preserve legacy TD3."""

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfmarl.agents.thermal_td3 import (
    TD3,
    Critic,
    TD3Config,
    channel_hash,
    delayed_actor_update_count,
)
from hfmarl.federation.representation import model_hash


def config(**changes):
    return TD3Config(hidden=8, batch=4, head_steps=3, rep_steps=3,
                     repair_steps=3, adapt_steps=4, **changes)


def learner(cfg=None):
    return TD3(9, np.array([0., 0.], np.float32),
               np.array([.25, .05], np.float32), 17, cfg or config())


def fill_replay(model):
    rng = np.random.default_rng(342)
    for index in range(16):
        model.remember(rng.normal(size=9).astype(np.float32),
                       rng.uniform(-1, 1, 2).astype(np.float32),
                       -float(index + 1), rng.normal(size=9).astype(np.float32),
                       index % 3 == 0)


def test_physical_critic_formula_and_normalized_equivalence_after_affine_reparameterization():
    physical = learner().critics[0]
    normalized = copy.deepcopy(physical)
    normalized.action_coordinates = "normalized"
    # z = lo + (a+1)*(hi-lo)/2. Absorb this affine map into
    # the first layer to express the SAME Q function in normalized coordinates.
    with torch.no_grad():
        weight = physical.trunk[0].weight[:, -2:]
        normalized.trunk[0].weight[:, -2:].copy_(weight * (physical.hi - physical.lo) * .5)
        normalized.trunk[0].bias.add_(weight @ ((physical.lo + physical.hi) * .5))
    obs = torch.randn((7, 9), generator=torch.Generator().manual_seed(551))
    action = torch.tensor([[-1., -1.], [0., 0.], [1., 1.], [.3, -.4],
                           [-.8, .9], [.7, -.6], [.1, .2]], requires_grad=True)
    power = physical.lo + (action + 1) * .5 * (physical.hi - physical.lo)
    expected = physical.head(physical.trunk(torch.cat([obs, power], dim=-1)))
    assert torch.equal(physical(obs, action), expected)
    torch.testing.assert_close(normalized(obs, action), expected, rtol=1e-6, atol=2e-8)
    physical_gradient = torch.autograd.grad(expected.sum(), action, retain_graph=True)[0]
    normalized_gradient = torch.autograd.grad(normalized(obs, action).sum(), action)[0]
    torch.testing.assert_close(normalized_gradient, physical_gradient, rtol=1e-5, atol=1e-8)


def test_normalized_input_removes_device_axis_scale_from_critic_action_gradient():
    physical = Critic(1, [0., 0.], [.25, .05], 1)
    with torch.no_grad():
        physical.trunk[0].weight.copy_(torch.tensor([[0., 1., 1.]]))
        physical.trunk[0].bias.fill_(3.)
        physical.trunk[2].weight.fill_(1.)
        physical.trunk[2].bias.zero_()
        physical.head.weight.fill_(1.)
        physical.head.bias.zero_()
    normalized = copy.deepcopy(physical)
    normalized.action_coordinates = "normalized"
    action = torch.tensor([[.2, -.2]], requires_grad=True)
    obs = torch.zeros((1, 1))
    physical_gradient = torch.autograd.grad(physical(obs, action).sum(), action)[0]
    normalized_gradient = torch.autograd.grad(normalized(obs, action).sum(), action)[0]
    torch.testing.assert_close(physical_gradient, torch.tensor([[.125, .025]]))
    assert torch.equal(normalized_gradient, torch.ones_like(action))
    torch.testing.assert_close(normalized_gradient / physical_gradient, torch.tensor([[8., 40.]]))


def test_coordinate_choice_preserves_initial_parameters_random_streams_and_representation_shapes():
    physical = learner()
    normalized = learner(config(critic_action_coordinates="normalized"))
    for name in ("actor", "critics", "target_actor", "target_critics"):
        for key, value in getattr(physical, name).state_dict().items():
            assert torch.equal(value, getattr(normalized, name).state_dict()[key])
    for name in ("rng", "replay_rng"):
        assert getattr(physical, name).bit_generator.state == getattr(normalized, name).bit_generator.state
    assert torch.equal(physical.torch_rng.get_state(), normalized.torch_rng.get_state())
    assert model_hash(physical.representation()) == model_hash(normalized.representation())
    assert channel_hash(physical.cfg) != channel_hash(normalized.cfg)
    assert all(c.action_coordinates == "normalized"
               for c in list(normalized.critics) + list(normalized.target_critics))
    for name, value in physical.representation().items():
        np.testing.assert_array_equal(value, normalized.representation()[name])
    normalized.install(physical.representation())
    assert all(c.action_coordinates == "normalized" for c in normalized.critics)


def test_coordinate_mode_is_explicit_in_serialized_configuration_and_old_defaults_restore():
    cfg = config()
    encoded = asdict(cfg)
    assert encoded["critic_action_coordinates"] == "physical"
    assert encoded["actor_warmup_updates"] == 0
    del encoded["critic_action_coordinates"], encoded["actor_warmup_updates"]
    assert TD3Config(**encoded) == cfg
    assert len({channel_hash(cfg),
                channel_hash(replace(cfg, critic_action_coordinates="normalized")),
                channel_hash(replace(cfg, actor_warmup_updates=300))}) == 3


def test_legacy_pickled_critic_without_coordinate_attribute_retains_physical_formula():
    critic = learner().critics[0]
    obs, action = torch.ones((2, 9)), torch.zeros((2, 2))
    before = critic(obs, action)
    del critic.action_coordinates
    assert torch.equal(critic(obs, action), before)


@pytest.mark.parametrize("value", [None, "", "watts", "normal", True])
def test_invalid_coordinate_mode_fails_before_training(value):
    with pytest.raises(ValueError, match="action coordinates"):
        config(critic_action_coordinates=value)
    with pytest.raises(ValueError, match="action coordinates"):
        Critic(9, [0., 0.], [.25, .05], 8, value)


@pytest.mark.parametrize("value", [-1, .5, 2., None, True, np.bool_(False), "300"])
def test_invalid_actor_warmup_fails_before_training(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        config(actor_warmup_updates=value)


@pytest.mark.parametrize("warmup", [3, 4])
def test_actor_burnin_learns_critics_and_delayed_targets_before_first_actor_update(warmup):
    model = learner(config(actor_warmup_updates=warmup))
    fill_replay(model)
    original_actor = copy.deepcopy(model.actor.state_dict())
    original_critics = copy.deepcopy(model.critics.state_dict())
    original_targets = copy.deepcopy(model.target_critics.state_dict())
    for update in range(1, warmup + 1):
        diagnostic = model.train_step()
        assert not diagnostic["actor_updated"]
        assert diagnostic["actor_loss"] is None
        assert diagnostic["actor_grad_norm"] is None
        assert not model.actor_opt.state
        assert all(torch.equal(value, model.actor.state_dict()[name])
                   for name, value in original_actor.items())
        assert all(torch.equal(value, model.target_actor.state_dict()[name])
                   for name, value in original_actor.items())
        if update == 1:
            assert all(torch.equal(value, model.target_critics.state_dict()[name])
                       for name, value in original_targets.items())
    assert any(not torch.equal(value, model.critics.state_dict()[name])
               for name, value in original_critics.items())
    assert any(not torch.equal(value, model.target_critics.state_dict()[name])
               for name, value in original_targets.items())
    first_actor_update = (warmup // model.cfg.policy_delay + 1) * model.cfg.policy_delay
    for update in range(warmup + 1, first_actor_update + 1):
        diagnostic = model.train_step()
        assert diagnostic["actor_updated"] == (update == first_actor_update)
    assert diagnostic["actor_loss"] is not None and diagnostic["actor_grad_norm"] > 0
    assert any(not torch.equal(value, model.actor.state_dict()[name])
               for name, value in original_actor.items())


def test_warmup_does_not_advance_without_replay():
    model = learner(config(actor_warmup_updates=4))
    assert model.train_step() is None
    assert model.steps == 0


@pytest.mark.parametrize("delay", [1, 2, 3])
@pytest.mark.parametrize("warmup", [0, 3, 4, 9])
def test_actor_update_count_matches_delayed_schedule_across_every_interval(delay, warmup):
    for start in range(10):
        for end in range(start, 12):
            expected = sum(update % delay == 0 and update > warmup
                           for update in range(start + 1, end + 1))
            assert delayed_actor_update_count(start, end, delay, warmup) == expected


@pytest.mark.parametrize("arguments", [(-1, 2, 2, 0), (3, 2, 2, 0), (0, 2, 0, 0),
                                        (0, 2, 2, -1), (0, 2., 2, 0), (False, 2, 2, 0)])
def test_actor_update_count_rejects_invalid_counter_intervals(arguments):
    with pytest.raises(ValueError, match="update-count"):
        delayed_actor_update_count(*arguments)


def test_repair_is_rejected_before_any_learning_or_rng_consumption_when_burnin_covers_adaptation():
    model = learner(config(actor_warmup_updates=4))
    fill_replay(model)
    actor = copy.deepcopy(model.actor.state_dict())
    critics = copy.deepcopy(model.critics.state_dict())
    replay_rng = copy.deepcopy(model.replay_rng.bit_generator.state)
    target_rng = model.torch_rng.get_state().clone()
    with pytest.raises(ValueError, match="actual actor learning"):
        model.repaired_fork()
    assert model.steps == 0
    assert not model.actor_opt.state and not model.critic_opt.state
    assert all(torch.equal(value, model.actor.state_dict()[name]) for name, value in actor.items())
    assert all(torch.equal(value, model.critics.state_dict()[name]) for name, value in critics.items())
    assert model.replay_rng.bit_generator.state == replay_rng
    assert torch.equal(model.torch_rng.get_state(), target_rng)


def test_repair_spanning_burnin_performs_and_accounts_only_actual_actor_updates():
    from hfmarl.experiments.coupled_control import repair_accounting
    from hfmarl.federation.repeated import ROLES

    model = learner(config(actor_warmup_updates=4))
    fill_replay(model)
    for _ in range(3):
        model.train_step()
    assert not model.actor_opt.state
    repaired = model.repaired_fork()
    assert (model.steps, repaired.steps) == (3, 7)
    # Updates 4 and 6 are delayed steps; only 6 follows burn-in.
    actor_steps = [float(state["step"]) for state in repaired.actor_opt.state.values()]
    assert actor_steps and all(step == 1 for step in actor_steps)
    assert not model.actor_opt.state
    beam_cfg = SimpleNamespace(repair_steps=3, batch=4)
    before = SimpleNamespace(heating=model, beam=SimpleNamespace(steps=3, cfg=beam_cfg))
    after = SimpleNamespace(heating=repaired, beam=SimpleNamespace(steps=7, cfg=beam_cfg))
    accounting = repair_accounting(before, after)
    assert accounting[ROLES[0]] == {
        "head_sgd_updates": 3, "critic_adaptation_updates": 4,
        "actor_adaptation_updates": 1, "entropy_temperature_updates": 0,
        "replay_samples": 28,
    }
    assert accounting[ROLES[1]]["actor_adaptation_updates"] == 4
    assert accounting[ROLES[1]]["entropy_temperature_updates"] == 4
    unchanged = repair_accounting(before, before, head_repaired=False)
    assert all(value == 0 for result in unchanged.values() for value in result.values())
