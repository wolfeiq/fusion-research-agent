"""Read-only counterfactual Q diagnostics for an already-trained thermal TD3.

These rollouts are measurements, never a policy-selection or repair procedure.
They compare online critics with one realized return under a frozen actor; target
policy smoothing and finite sampling mean this is not an exact Q-function oracle.
"""
import copy

import numpy as np
import torch


ECRH_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5)
AUX_FRACTION = 0.125


def _actor_action(actor, observation):
    parameter = next(actor.parameters())
    x = torch.as_tensor(observation, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        action = actor(x.unsqueeze(0))[0].detach().cpu().numpy()
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Q probes require a finite two-actuator policy")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _critic_prediction(critic, observation, action):
    parameter = next(critic.parameters())
    with torch.enable_grad():
        x = torch.as_tensor(observation, dtype=parameter.dtype,
                            device=parameter.device).unsqueeze(0)
        u = torch.tensor(action, dtype=parameter.dtype, device=parameter.device,
                         requires_grad=True).unsqueeze(0)
        prediction = critic(x, u).squeeze()
        if prediction.ndim != 0 or not torch.isfinite(prediction):
            raise FloatingPointError("Nonfinite or nonscalar initial critic prediction")
        gradient, = torch.autograd.grad(prediction, u)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite initial critic action derivative")
    return float(prediction.detach()), gradient[0].detach().cpu().tolist()


def _replay_support(learner):
    if not learner.replay:
        return {"transitions": 0, "fraction_min": None, "fraction_max": None,
                "meaning": "coordinate_bounds_only_not_joint_state_action_coverage"}
    actions = np.asarray([row[1] for row in learner.replay], dtype=float)
    if actions.shape != (len(learner.replay), 2) or not np.isfinite(actions).all():
        raise ValueError("Replay contains invalid two-actuator actions")
    fractions = (actions + 1.0) * 0.5
    return {"transitions": len(actions), "fraction_min": fractions.min(0).tolist(),
            "fraction_max": fractions.max(0).tolist(),
            "meaning": "coordinate_bounds_only_not_joint_state_action_coverage"}


def evaluate_q_probes(study, name, learner, physical_seed):
    """Return six JSON-safe rollout records without modifying the learner.

    Every episode starts with the same reset seed. Five probe actions fix auxiliary
    heating at 12.5% while varying ECRH; the sixth uses both actor-selected commands.
    After that first action, the unchanged deterministic actor controls the episode.
    Fractions refer to actuator endpoints, not a common physical MW scale.

    Source actor/critics, target networks, replay, optimizers, gradients and random
    streams are untouched. Environment state is reset and stepped as in ordinary
    evaluation; its original RNG object is restored, including on an exception.
    """
    gamma = float(learner.cfg.gamma)
    reward_scale = float(learner.cfg.reward_scale)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Q probes require a finite discount in [0, 1]")
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Q probes require a finite positive reward scale")
    if len(learner.critics) != 2:
        raise ValueError("Q probes require two critic branches")

    # Copies also protect mutable module buffers and training flags. No learner
    # forward method, train_step, replay sampler or random generator is called.
    actor = copy.deepcopy(learner.actor).eval()
    critics = copy.deepcopy(learner.critics).eval()
    for model in [actor, critics]:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    support = _replay_support(learner)
    env, adapter = study.env(name)
    previous_rng = env.rng
    initial_observation = None
    records = []
    try:
        for index in range(len(ECRH_FRACTIONS) + 1):
            env.rng = np.random.default_rng(int(physical_seed))
            raw, _ = env.reset()
            observation = adapter.observe(env, raw)
            if not np.isfinite(observation).all():
                raise ValueError("Nonfinite initial observation")
            if initial_observation is None:
                initial_observation = observation.copy()
            elif not np.array_equal(initial_observation, observation):
                raise RuntimeError("Q probes did not reset to identical initial observations")
            actor_action = _actor_action(actor, observation)
            if index < len(ECRH_FRACTIONS):
                first_action = np.array([2 * AUX_FRACTION - 1,
                                         2 * ECRH_FRACTIONS[index] - 1], np.float32)
                label = f"fixed_ecrh_{ECRH_FRACTIONS[index]:g}"
            else:
                first_action = actor_action.copy()
                label = "actor"
            predictions = [_critic_prediction(critic, observation, first_action)
                           for critic in critics]
            fractions = (first_action.astype(float) + 1.0) * 0.5
            within_bounds = None
            if support["transitions"]:
                within_bounds = bool(np.all(fractions >= support["fraction_min"])
                                     and np.all(fractions <= support["fraction_max"]))
            total_return, cost, steps = 0.0, 0.0, 0
            events, rewards = set(), []
            while True:
                action = first_action if steps == 0 else _actor_action(actor, observation)
                raw, reward, terminated, truncated, info = env.step(action)
                try:
                    successor = adapter.observe(env, raw)
                    if not np.isfinite(successor).all():
                        raise ValueError("Nonfinite successor observation")
                except ValueError:
                    if not terminated:
                        raise
                    # A terminal successor is never a bootstrap or actor input.
                    successor = np.zeros_like(observation)
                if not np.isfinite(reward):
                    raise FloatingPointError("Nonfinite environment reward")
                reward = float(reward)
                total_return += gamma ** steps * (reward / reward_scale)
                cost -= reward
                if not np.isfinite(total_return) or not np.isfinite(cost):
                    raise FloatingPointError("Nonfinite accumulated probe return or cost")
                steps += 1
                rewards.append(reward)
                events.update(info.get("violations", ()))
                observation = successor
                if terminated or truncated:
                    break
                if steps > env.task.steps_per_shot + 1:
                    raise RuntimeError("TORAX failed to end the declared horizon")
            q1, q2 = (prediction[0] for prediction in predictions)
            records.append({
                "probe": label, "device": name, "physical_seed": int(physical_seed),
                "initial_observation": initial_observation.tolist(),
                "action": first_action.tolist(), "action_fraction": fractions.tolist(),
                "actor_action": actor_action.tolist(),
                "actor_fraction": ((actor_action.astype(float) + 1) * 0.5).tolist(),
                "replay_support": copy.deepcopy(support),
                "first_action_in_replay_coordinate_bounds": within_bounds,
                "q1": q1, "q2": q2, "q_min": min(q1, q2),
                "dq1_du": predictions[0][1], "dq2_du": predictions[1][1],
                "derivative_units": "Q_per_unit_command[-1,1]; order_aux_heat_ecrh",
                "discounted_return": total_return,
                "q1_minus_return": q1 - total_return,
                "q2_minus_return": q2 - total_return,
                "q_min_minus_return": min(q1, q2) - total_return,
                "gamma": gamma, "reward_scale": reward_scale,
                "cost": cost, "events": sorted(events), "steps": steps,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "rewards": rewards,
                "purpose": "read_only_value_and_action_ranking_diagnostic_not_policy_selection",
            })
    finally:
        env.rng = previous_rng
    return records
