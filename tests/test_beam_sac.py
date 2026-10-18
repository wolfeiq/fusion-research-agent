"""SAC mathematics, training, isolation and exact local checkpoint recovery."""
import copy
import io
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install the rl extra for SAC tests")

from hfmarl.agents.beam_sac import SAC, SACConfig, GaussianActor, channel_hash
from hfmarl.federation.representation import UniformRepresentationServer, model_hash


def learner(seed=17, fill=True, action_dim=2):
    config = SACConfig(hidden=12, batch=4, replay_capacity=16, head_steps=2,
                       rep_steps=2, repair_steps=2, adapt_steps=2)
    result = SAC(5, seed, config, action_dim=action_dim)
    if fill:
        rng = np.random.default_rng(414)
        for step in range(22):
            result.remember(rng.normal(size=5).astype(np.float32),
                            rng.uniform(-1, 1, action_dim).astype(np.float32),
                            -float(step + 1), rng.normal(size=5).astype(np.float32),
                            step % 3 == 0)
    return result


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def test_log_probability_matches_tanh_transformed_normal_and_reparameterizes():
    policy = GaussianActor(3, 8)
    observation = torch.randn(7, 3)
    generator = torch.Generator().manual_seed(91)
    action, log_probability = policy.sample(observation, generator)
    mean, log_std = policy.distribution_parameters(observation)
    base = torch.distributions.Normal(mean, log_std.exp())
    oracle = torch.distributions.TransformedDistribution(
        base, [torch.distributions.TanhTransform(cache_size=1)])
    expected = oracle.log_prob(action).sum(-1, keepdim=True)
    assert log_probability.shape == (7, 1)
    torch.testing.assert_close(log_probability, expected, atol=2e-6, rtol=2e-6)
    (action.square().sum() + log_probability.sum()).backward()
    assert policy.mean.weight.grad.abs().sum() > 0
    assert policy.log_std.weight.grad.abs().sum() > 0


def test_saturated_policy_has_finite_log_density_and_gradients():
    policy = GaussianActor(3, 8)
    with torch.no_grad():
        policy.mean.bias.copy_(torch.tensor([100., -100.]))
    action, log_probability = policy.sample(torch.ones(3, 3), torch.Generator().manual_seed(1))
    assert torch.isfinite(log_probability).all()
    assert torch.all(action[:, 0] == 1) and torch.all(action[:, 1] == -1)
    log_probability.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_terminal_labels_stop_bootstrap_but_truncation_labels_do_not():
    model = learner()
    reward = torch.tensor([[-2.], [-2.]])
    nxt = torch.ones(2, 5)
    terminal = torch.tensor([[1.], [0.]])
    reference = copy.deepcopy(model)
    with torch.no_grad():
        action, log_probability = reference.actor.sample(nxt, reference.torch_rng)
        expected_soft_q = torch.minimum(*(q(nxt, action) for q in reference.target_critics))
        expected_soft_q -= reference.alpha.detach() * log_probability
    actual = model.labels(reward, nxt, terminal)
    assert actual[0].item() == -2.
    torch.testing.assert_close(actual[1], reward[1] + model.cfg.gamma * expected_soft_q[1])
    assert actual[1].item() != -2.
    assert not actual.requires_grad


def test_updates_change_actor_both_critics_targets_and_temperature():
    model = learner()
    before = model.state_dict()
    metrics = model.train_step()
    assert metrics["updates"] == 1 and metrics["actor_updated"]
    assert metrics["actor_grad_norm"] > 0 and metrics["critic_grad_norm"] > 0
    assert metrics["q1_loss"] > 0 and metrics["q2_loss"] > 0
    for name in ("actor", "critics", "target_critics"):
        assert any(not torch.equal(value, getattr(model, name).state_dict()[key])
                   for key, value in before[name].items())
    for branch in range(2):
        assert not torch.equal(before["critics"][f"{branch}.head.weight"],
                               model.critics[branch].head.weight)
    assert not torch.equal(before["log_alpha"], model.log_alpha)
    assert all(np.isfinite(value) for value in metrics.values())
    json.dumps(metrics, allow_nan=False)
    for name, value in model.target_critics.state_dict().items():
        expected = before["target_critics"][name].lerp(model.critics.state_dict()[name], model.cfg.tau)
        torch.testing.assert_close(value, expected)
    assert all(p.grad is None for p in model.target_critics.parameters())


def test_actor_evaluation_and_replay_do_not_consume_each_others_random_streams():
    model = learner()
    reference = copy.deepcopy(model)
    observation = np.zeros(5, np.float32)
    expected = model.action(observation)
    for _ in range(4):
        np.testing.assert_array_equal(model.action(observation), expected)
        model.batch()
        model.frozen_dataset(1)
    for _ in range(4):
        np.testing.assert_array_equal(model.action(observation, explore=True),
                                      reference.action(observation, explore=True))
    reference = copy.deepcopy(model)
    for _ in range(3):
        model.action(observation, explore=True)
    equal_tree(model.batch(), reference.batch())
    equal_tree(model.rng.bit_generator.state, reference.rng.bit_generator.state)


def test_constructor_and_training_do_not_consume_global_torch_rng():
    before = torch.random.get_rng_state().clone()
    model = learner()
    model.train_step()
    model.action(np.zeros(5, np.float32), explore=True)
    assert torch.equal(before, torch.random.get_rng_state())


def test_checkpoint_resumes_replay_wraparound_models_optimizers_and_rngs():
    model = learner()
    observation = np.ones(5, np.float32)
    for _ in range(3):
        model.train_step()
        model.action(observation, explore=True)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    resumed = SAC.load(checkpoint)
    equal_tree(model.state_dict(), resumed.state_dict())
    for _ in range(4):
        expected_action = model.action(observation, explore=True)
        np.testing.assert_array_equal(expected_action, resumed.action(observation, explore=True))
        assert model.train_step() == resumed.train_step()
        for item in (model, resumed):
            item.remember(observation, expected_action, -.5, observation * .9, False)
        equal_tree(model.state_dict(), resumed.state_dict())
    assert model.cursor == model.env_steps % model.cfg.replay_capacity


def test_insufficient_replay_does_not_change_any_state():
    model = learner(fill=False)
    before = model.state_dict()
    assert model.train_step() is None
    equal_tree(before, model.state_dict())


def test_atomic_file_checkpoint_roundtrip(tmp_path):
    model = learner()
    model.train_step()
    path = tmp_path / "beam.pt"
    model.save(path)
    equal_tree(model.state_dict(), SAC.load(path).state_dict())
    assert not path.with_suffix(".pt.tmp").exists()


def test_representation_job_exports_only_common_base_trunks_without_local_mutation():
    first, second = learner(17), learner(29)
    base = first.representation()
    equal_tree(base, second.representation())
    before = first.state_dict()
    schema = channel_hash(first.cfg, "fixed-physical-observations-v1")
    update = first.contribution(base, "a", "a:0", schema, "revision-a")
    assert update.base_hash == model_hash(base)
    assert set(update.tensors) == set(base)
    assert all("head" not in key for key in update.tensors)
    assert any(np.any(delta != 0) for delta in update.tensors.values())
    equal_tree(before, first.state_dict())
    server = UniformRepresentationServer(base, ["a", "b"], schema,
                                         {"a": "revision-a", "b": "revision-b"})
    proposal = server.aggregate([update, second.contribution(base, "b", "b:0", schema, "revision-b")])
    actor_before, head_before = copy.deepcopy(first.actor.state_dict()), copy.deepcopy(first.critics[0].head.state_dict())
    first.install(proposal)
    equal_tree(actor_before, first.actor.state_dict())
    equal_tree(head_before, first.critics[0].head.state_dict())
    candidate, reset = second.repaired_fork(proposal), second.repaired_fork()
    assert candidate.steps == reset.steps == second.steps + second.cfg.adapt_steps
    assert any(not torch.equal(value, candidate.actor.state_dict()[name])
               for name, value in second.actor.state_dict().items())


def test_installing_common_base_does_not_rebase_frozen_bellman_targets():
    model = learner()
    reference = copy.deepcopy(model)
    changed_base = {name: value + .5 for name, value in model.representation().items()}
    model.install(changed_base)
    original_labels, original_scale = reference.frozen_dataset(3)
    new_labels, new_scale = model.frozen_dataset(3)
    equal_tree(original_labels, new_labels)
    assert original_scale == new_scale
    equal_tree(reference.target_critics.state_dict(), model.target_critics.state_dict())
    equal_tree(reference.actor.state_dict(), model.actor.state_dict())
    equal_tree(reference.log_alpha, model.log_alpha)


def test_joint_action_baseline_is_supported_and_has_distinct_channel_schema():
    model = learner(action_dim=4)
    assert model.action(np.ones(5, np.float32)).shape == (4,)
    assert model.target_entropy == -4.
    assert model.train_step()["actor_updated"]
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    equal_tree(model.state_dict(), SAC.load(checkpoint).state_dict())
    with pytest.raises(ValueError, match="action_schema"):
        channel_hash(model.cfg, "obs", action_dim=4)
    assert channel_hash(model.cfg, "obs") != channel_hash(
        model.cfg, "obs", action_dim=4, action_schema="aux,budget,radius,fraction")


@pytest.mark.parametrize("bad", [
    {"gamma": float("nan")}, {"tau": 2}, {"initial_alpha": 0}, {"reward_scale": 0},
    {"batch": 0}, {"replay_capacity": 2}, {"hidden": True}, {"mix": -1},
])
def test_configuration_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        SACConfig(**bad)


def test_invalid_install_or_checkpoint_does_not_mutate_live_learner():
    model = learner()
    before = model.state_dict()
    values = model.representation()
    next(iter(values.values())).fill(np.nan)
    with pytest.raises(ValueError):
        model.install(values)
    equal_tree(before, model.state_dict())
    corrupt = copy.deepcopy(before)
    corrupt["cursor"] += 1
    with pytest.raises(ValueError, match="counters"):
        model.load_state_dict(corrupt)
    equal_tree(before, model.state_dict())


@pytest.mark.parametrize("action,reward,terminated", [
    ([0., 2.], 1., False), ([0., float("nan")], 1., False),
    ([0., 0.], float("inf"), False), ([0., 0.], 1., .5),
])
def test_invalid_experience_is_rejected(action, reward, terminated):
    model = learner(fill=False)
    with pytest.raises(ValueError):
        model.remember(np.zeros(5), action, reward, np.zeros(5), terminated)
    assert len(model.replay) == model.env_steps == 0
