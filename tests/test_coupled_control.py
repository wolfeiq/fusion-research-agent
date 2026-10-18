"""Causal replay and real optimizer checks; this toy is not TORAX evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmarl.agents.beam_sac import SAC, SACConfig
from hfmarl.agents.thermal_td3 import TD3Config
from hfmarl.experiments.coupled_control import LearnerPair, episode


class TinyEnv:
    task = SimpleNamespace(steps_per_shot=5)

    def reset(self):
        self.t = 0
        self.x = np.zeros(37, np.float32)
        return self.x.copy(), {}

    def step(self, action):
        assert action.shape == (4,) and np.isfinite(action).all()
        self.x[:4] += action * .1
        self.t += 1
        self.x[4] = self.t / 5
        return self.x.copy(), -float(np.square(self.x[:4] - .2).sum()), False, self.t == 5, {}


def small_pair():
    return LearnerPair(7, TD3Config(batch=2, hidden=8, reward_scale=1.),
                       SACConfig(batch=2, hidden=8), [1., 1.])


def test_next_budget_in_sac_replay_is_the_budget_actually_used():
    pair = small_pair()
    result = episode(TinyEnv(), pair, 19, train=True, active="both")
    assert result["complete"] and result["windows"] == 5
    assert result["optimizer_updates"] == {"heating": 4, "beam": 4, "joint": 0}
    for index, transition in enumerate(pair.beam.replay):
        observation, action, _, nxt, terminal = transition
        np.testing.assert_array_equal(observation[-2:], result["trace"][index]["action"][:2])
        np.testing.assert_array_equal(action, result["trace"][index]["action"][2:])
        if index < 4:
            np.testing.assert_array_equal(nxt[-2:], result["trace"][index + 1]["action"][:2])
            assert terminal == 0
        else:
            assert terminal == 1


def test_frozen_heating_gets_replay_but_no_optimizer_updates():
    pair = small_pair()
    before = [p.detach().clone() for p in pair.heating.actor.parameters()]
    result = episode(TinyEnv(), pair, 19, train=True, active="beam")
    assert result["optimizer_updates"]["heating"] == 0
    assert len(pair.heating.replay) == 5
    assert all((a == b).all() for a, b in zip(before, pair.heating.actor.parameters()))
    assert result["optimizer_updates"]["beam"] == 4


def test_joint_controller_controls_same_four_actions_and_learns():
    learner = SAC(37, 7, SACConfig(hidden=8, batch=2), action_dim=4)
    result = episode(TinyEnv(), learner, 19, train=True)
    assert result["optimizer_updates"]["joint"] == 4
    assert learner.replay[-1][-1] == 1
    assert result["complete"]


def test_evaluation_does_not_train_or_mutate_replay():
    pair = small_pair()
    first = episode(TinyEnv(), pair, 19)
    second = episode(TinyEnv(), pair, 19)
    assert first == second
    assert not pair.heating.replay and not pair.beam.replay


def test_incomplete_or_invalid_episode_is_not_success():
    class Broken(TinyEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, False, False, info
    with pytest.raises(RuntimeError, match="horizon"):
        episode(Broken(), small_pair(), 19)
