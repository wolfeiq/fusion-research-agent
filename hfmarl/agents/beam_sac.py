"""Local SAC for two normalized microwave-deposition commands.

The environment owns the conversion from [-1, 1] to deposition radius and
fraction of the heating agent's EC power budget. This module does not model
mirror angles, magnetic islands or NTM suppression. Entropy is measured in the
normalized action coordinates, whose affine action scale is exactly one.

Only the twin critics' trunks are eligible for the separate beam-SAC federation
channel. Actors, entropy temperature, heads, targets, replay and optimizers stay
local. The deliberately small networks currently execute on CPU.
"""
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hfmarl.federation.representation import RepresentationDelta, model_hash


@dataclass(frozen=True)
class SACConfig:
    hidden: int = 64
    batch: int = 256
    lr: float = 3e-4
    actor_lr: float | None = None
    alpha_lr: float = 3e-4
    initial_alpha: float = .2
    target_entropy: float | None = None
    gamma: float = .99
    tau: float = .005
    reward_scale: float = 1.0
    replay_capacity: int = 100000
    rep_lr: float = 1e-3
    head_steps: int = 100
    rep_steps: int = 100
    repair_steps: int = 100
    adapt_steps: int = 100
    mix: float = .25

    def __post_init__(self):
        for name in ("hidden", "batch", "replay_capacity", "head_steps",
                     "rep_steps", "repair_steps", "adapt_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.replay_capacity < self.batch:
            raise ValueError("replay_capacity must be at least batch")
        for name in ("lr", "alpha_lr", "initial_alpha", "reward_scale", "rep_lr"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.actor_lr is not None and (
                not np.isfinite(self.actor_lr) or self.actor_lr <= 0):
            raise ValueError("actor_lr must be finite and positive")
        for name in ("gamma", "tau", "mix"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.target_entropy is not None and not np.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")


class GaussianActor(nn.Module):
    """Reparameterized Gaussian policy, squashed into the normalized action box."""

    def __init__(self, obs_dim, width, action_dim=2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)
        nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -1.)

    def distribution_parameters(self, obs):
        features = self.trunk(obs)
        return self.mean(features), self.log_std(features).clamp(-20., 2.)

    def forward(self, obs):
        """Deterministic evaluation uses tanh(mean), not a random draw."""
        mean, _ = self.distribution_parameters(obs)
        return torch.tanh(mean)

    def sample(self, obs, generator):
        mean, log_std = self.distribution_parameters(obs)
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype,
                            device=mean.device)
        pre_tanh = mean + log_std.exp() * noise
        action = torch.tanh(pre_tanh)
        log_normal = -.5 * (noise.square() + 2 * log_std + math.log(2 * math.pi))
        # Stable log(1-tanh(u)^2), including saturated actions. The affine
        # transform to [-1,1] has scale=1, so its log-Jacobian is zero.
        log_jacobian = 2 * (math.log(2.) - pre_tanh - F.softplus(-2 * pre_tanh))
        return action, (log_normal - log_jacobian).sum(dim=-1, keepdim=True)


class BeamCritic(nn.Module):
    def __init__(self, obs_dim, width, action_dim=2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width, 1)

    def forward(self, obs, action):
        return self.head(self.trunk(torch.cat((obs, action), dim=-1)))


class SAC:
    """SAC with isolated action, replay and learning random-number streams."""

    checkpoint_version = 1
    algorithm = "beam-sac-v1"

    def __init__(self, obs_dim, seed, cfg=SACConfig(), representation_seed=2718,
                 action_dim=2):
        if isinstance(obs_dim, bool) or not isinstance(obs_dim, int) or obs_dim <= 0:
            raise ValueError("obs_dim must be a positive integer")
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        self.obs_dim, self.action_dim, self.seed = obs_dim, action_dim, int(seed)
        self.cfg, self.representation_seed = cfg, int(representation_seed)
        self.target_entropy = -float(action_dim) if cfg.target_entropy is None else cfg.target_entropy
        self.rng = np.random.default_rng(seed)  # external warm-up actions
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5245504C]))
        self.action_rng = torch.Generator().manual_seed(seed)
        self.torch_rng = torch.Generator().manual_seed(seed + 0x534143)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.actor = GaussianActor(obs_dim, cfg.hidden, action_dim)
            self.critics = nn.ModuleList()
            for branch in range(2):
                torch.manual_seed(representation_seed + branch)
                critic = BeamCritic(obs_dim, cfg.hidden, action_dim)
                torch.manual_seed(seed + 100 + branch)
                critic.head.reset_parameters()
                self.critics.append(critic)
        self.target_critics = copy.deepcopy(self.critics)
        self.target_critics.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha)))
        actor_lr = cfg.lr if cfg.actor_lr is None else cfg.actor_lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.replay = []
        self.cursor = self.steps = self.env_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _observation(self, obs):
        value = np.asarray(obs, dtype=np.float32)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise ValueError("Observation must match obs_dim and be finite")
        return value

    def action(self, obs, explore=False):
        value = torch.from_numpy(self._observation(obs)).unsqueeze(0)
        with torch.no_grad():
            action = (self.actor.sample(value, self.action_rng)[0]
                      if explore else self.actor(value))
        result = action[0].numpy().copy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite SAC action")
        return result

    def remember(self, obs, action, reward, nxt, terminated):
        observation, next_observation = self._observation(obs), self._observation(nxt)
        action = np.asarray(action, dtype=np.float32)
        if (action.shape != (self.action_dim,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1)):
            raise ValueError("Actions must match action_dim and lie in [-1,1]")
        if not np.isscalar(reward) or not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("terminated must be boolean; truncations must bootstrap")
        row = (observation.copy(), action.copy(), float(reward) / self.cfg.reward_scale,
               next_observation.copy(), float(terminated))
        if len(self.replay) < self.cfg.replay_capacity:
            self.replay.append(row)
        else:
            self.replay[self.cursor] = row
        self.cursor = (self.cursor + 1) % self.cfg.replay_capacity
        self.env_steps += 1

    def batch(self):
        if not self.replay:
            raise ValueError("Cannot sample empty replay")
        ids = self.replay_rng.integers(len(self.replay), size=self.cfg.batch)
        values = zip(*(self.replay[i] for i in ids))
        obs, actions, rewards, nxt, terminals = [
            torch.tensor(np.array(part), dtype=torch.float32) for part in values]
        return obs, actions, rewards[:, None], nxt, terminals[:, None]

    @torch.no_grad()
    def labels(self, rewards, nxt, terminals):
        actions, log_prob = self.actor.sample(nxt, self.torch_rng)
        soft_value = torch.minimum(*(c(nxt, actions) for c in self.target_critics))
        soft_value = soft_value - self.alpha.detach() * log_prob
        return rewards + self.cfg.gamma * (1 - terminals) * soft_value

    @staticmethod
    def _optimize(loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SAC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing or nonfinite SAC gradient")
        norm = float(sum(g.detach().double().square().sum() for g in gradients).sqrt())
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Nonfinite SAC parameters")
        return norm

    def train_step(self):
        if len(self.replay) < self.cfg.batch:
            return None
        obs, action, reward, nxt, terminal = self.batch()
        targets = self.labels(reward, nxt, terminal)
        predictions = [critic(obs, action) for critic in self.critics]
        errors = [q - targets for q in predictions]
        losses = [e.square().mean() for e in errors]
        critic_loss = sum(losses)
        critic_norm = self._optimize(critic_loss, self.critic_opt)
        self.critics.requires_grad_(False)
        try:
            policy_action, log_prob = self.actor.sample(obs, self.torch_rng)
            q = torch.minimum(*(critic(obs, policy_action) for critic in self.critics))
            actor_loss = (self.alpha.detach() * log_prob - q).mean()
            actor_norm = self._optimize(actor_loss, self.actor_opt)
        finally:
            self.critics.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        alpha_norm = self._optimize(alpha_loss, self.alpha_opt)
        if not torch.isfinite(self.alpha) or self.alpha <= 0:
            raise FloatingPointError("Invalid SAC entropy temperature")
        with torch.no_grad():
            for online, target in zip(self.critics.parameters(), self.target_critics.parameters()):
                target.lerp_(online, self.cfg.tau)
        self.steps += 1
        return {
            "updates": self.steps, "actor_updated": True,
            "critic_loss": float(critic_loss.detach()),
            "q1_loss": float(losses[0].detach()), "q2_loss": float(losses[1].detach()),
            "td_abs_mean": float(torch.stack([e.detach().abs().mean() for e in errors]).mean()),
            "q1_mean": float(predictions[0].detach().mean()),
            "q2_mean": float(predictions[1].detach().mean()),
            "target_mean": float(targets.mean()), "target_min": float(targets.min()),
            "target_max": float(targets.max()), "actor_loss": float(actor_loss.detach()),
            "critic_grad_norm": critic_norm, "actor_grad_norm": actor_norm,
            "alpha_loss": float(alpha_loss.detach()), "alpha_grad_norm": alpha_norm,
            "alpha": float(self.alpha.detach()), "entropy": -float(log_prob.detach().mean()),
        }

    def representation(self):
        return {f"q{k}.{name}": value.detach().numpy().copy()
                for k, critic in enumerate(self.critics)
                for name, value in critic.trunk.state_dict().items()}

    def install(self, values, mix=1.):
        current = self.representation()
        if set(values) != set(current) or not np.isfinite(mix) or not 0 <= mix <= 1:
            raise ValueError("Invalid SAC representation")
        for name, value in values.items():
            if (not isinstance(value, np.ndarray) or value.shape != current[name].shape
                    or value.dtype != current[name].dtype or not np.isfinite(value).all()):
                raise ValueError("Invalid SAC representation tensor")
        for k, critic in enumerate(self.critics):
            critic.trunk.load_state_dict({
                name: torch.from_numpy((1 - mix) * current[f"q{k}.{name}"]
                                       + mix * values[f"q{k}.{name}"])
                for name in critic.trunk.state_dict()})

    def frozen_dataset(self, batches):
        if batches <= 0:
            raise ValueError("Need at least one frozen replay batch")
        data = []
        for _ in range(batches):
            obs, action, reward, nxt, terminal = self.batch()
            data.append((obs, action, self.labels(reward, nxt, terminal)))
        scale = max(float(torch.cat([row[2] for row in data]).square().mean().sqrt()), 1.)
        return data, scale

    def fit_parts(self, data, scale, part, count, lr):
        if part not in {"head", "trunk"} or not data or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid SAC representation fitting job")
        self.critics.requires_grad_(False)
        for critic in self.critics:
            getattr(critic, part).requires_grad_(True)
        optimizer = torch.optim.SGD(
            [p for c in self.critics for p in getattr(c, part).parameters()], lr=lr)
        try:
            for step in range(count):
                obs, action, target = data[step % len(data)]
                loss = sum(((critic(obs, action) - target) / scale).square().mean()
                           for critic in self.critics)
                self._optimize(loss, optimizer)
        finally:
            self.critics.requires_grad_(True)

    def contribution(self, base, source, update_id, schema, revision):
        """Frozen-policy common-base job; the live learner is never mutated."""
        if len(self.replay) < self.cfg.batch:
            raise ValueError("Insufficient training replay")
        job = copy.deepcopy(self)
        job.install(base)
        data, scale = job.frozen_dataset(max(self.cfg.head_steps, self.cfg.rep_steps))
        job.fit_parts(data, scale, "head", self.cfg.head_steps, self.cfg.rep_lr)
        job.fit_parts(data, scale, "trunk", self.cfg.rep_steps, self.cfg.rep_lr)
        delta = {name: value - base[name] for name, value in job.representation().items()}
        return RepresentationDelta(update_id, source, model_hash(base), schema, revision, delta)

    def repaired_fork(self, proposal=None):
        if len(self.replay) < self.cfg.batch:
            raise ValueError("Transfer gate requires enough replay and actor adaptation")
        job = copy.deepcopy(self)
        if proposal is not None:
            job.install(proposal, self.cfg.mix)
        data, scale = job.frozen_dataset(self.cfg.repair_steps)
        job.fit_parts(data, scale, "head", self.cfg.repair_steps, self.cfg.rep_lr)
        job.target_critics = copy.deepcopy(job.critics).requires_grad_(False)
        job.critic_opt = torch.optim.Adam(job.critics.parameters(), lr=self.cfg.lr)
        for _ in range(self.cfg.adapt_steps):
            job.train_step()
        return job

    def state_dict(self):
        """Complete local state, including replay and all independent RNGs."""
        replay = [(torch.from_numpy(o.copy()), torch.from_numpy(a.copy()), r,
                   torch.from_numpy(n.copy()), terminal) for o, a, r, n, terminal in self.replay]
        state = {
            "version": self.checkpoint_version, "algorithm": self.algorithm,
            "obs_dim": self.obs_dim, "action_dim": self.action_dim, "seed": self.seed,
            "representation_seed": self.representation_seed, "config": asdict(self.cfg),
            "actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(),
            "alpha_optimizer": self.alpha_opt.state_dict(),
            "replay": replay, "cursor": self.cursor, "steps": self.steps,
            "env_steps": self.env_steps, "rng": self.rng.bit_generator.state,
            "replay_rng": self.replay_rng.bit_generator.state,
            "action_rng": self.action_rng.get_state(), "torch_rng": self.torch_rng.get_state(),
        }
        return copy.deepcopy(state)

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != cls.checkpoint_version or state.get("algorithm") != cls.algorithm:
            raise ValueError("Unsupported SAC checkpoint")
        restored = cls(state["obs_dim"], state["seed"], SACConfig(**state["config"]),
                       state["representation_seed"], state["action_dim"])
        for name in ("actor", "critics", "target_critics"):
            model = getattr(restored, name)
            model.load_state_dict(state[name], strict=True)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite checkpoint parameters")
        with torch.no_grad():
            restored.log_alpha.copy_(state["log_alpha"])
        if not torch.isfinite(restored.alpha) or restored.alpha <= 0:
            raise ValueError("Invalid checkpoint entropy temperature")
        for name in ("actor", "critic", "alpha"):
            getattr(restored, f"{name}_opt").load_state_dict(state[f"{name}_optimizer"])
        if len(state["replay"]) > restored.cfg.replay_capacity:
            raise ValueError("Replay exceeds checkpoint capacity")
        for obs, action, reward, nxt, terminal in state["replay"]:
            if terminal not in (0., 1.) or not np.isfinite(reward):
                raise ValueError("Invalid replay reward or terminal flag")
            observation, next_observation = restored._observation(obs.numpy()), restored._observation(nxt.numpy())
            command = action.numpy()
            if command.shape != (restored.action_dim,) or not np.isfinite(command).all() or np.any(np.abs(command) > 1):
                raise ValueError("Invalid replay action")
            restored.replay.append((observation.copy(), command.copy(), float(reward),
                                    next_observation.copy(), float(terminal)))
        for name in ("cursor", "steps", "env_steps"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid checkpoint counter")
            setattr(restored, name, value)
        if (restored.cursor != restored.env_steps % restored.cfg.replay_capacity
                or len(restored.replay) != min(restored.env_steps, restored.cfg.replay_capacity)):
            raise ValueError("Inconsistent checkpoint replay counters")
        restored.rng.bit_generator.state = copy.deepcopy(state["rng"])
        restored.replay_rng.bit_generator.state = copy.deepcopy(state["replay_rng"])
        restored.action_rng.set_state(state["action_rng"])
        restored.torch_rng.set_state(state["torch_rng"])
        return restored

    def load_state_dict(self, state):
        restored = type(self).from_state_dict(state)
        if (restored.obs_dim != self.obs_dim or restored.action_dim != self.action_dim
                or restored.cfg != self.cfg
                or restored.representation_seed != self.representation_seed):
            raise ValueError("Checkpoint architecture or configuration mismatch")
        self.__dict__.update(restored.__dict__)

    def save(self, path):
        """Write a complete checkpoint atomically on the local filesystem."""
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def channel_hash(cfg, observation_schema, action_dim=2, action_schema=None):
    """Callers must register matching physical observation/action semantics."""
    if not isinstance(observation_schema, str) or not observation_schema:
        raise ValueError("A nonempty physical observation schema is required")
    if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
        raise ValueError("action_dim must be a positive integer")
    if action_dim != 2 and not action_schema:
        raise ValueError("Non-beam action spaces require an explicit action_schema")
    payload = {"algorithm": SAC.algorithm, "observations": observation_schema,
               "action_dim": action_dim,
               "actions": action_schema or "normalized:deposition-radius,allocated-ec-budget-fraction",
               "sac": asdict(cfg)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
