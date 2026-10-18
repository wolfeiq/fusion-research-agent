"""Local TD3 and disposable shared-critic jobs for the thermal pilot.

No actor, local head, replay, target or optimizer is federated. CPU torch keeps
small networks deterministic and leaves accelerator selection to TORAX.
"""
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import torch
from torch import nn

from hfmarl.federation.representation import RepresentationDelta, model_hash


@dataclass(frozen=True)
class TD3Config:
    hidden: int = 64
    batch: int = 256
    lr: float = 3e-4
    rep_lr: float = 1e-3
    gamma: float = .99
    tau: float = .005
    policy_delay: int = 2
    target_noise: float = .2
    noise_clip: float = .5
    exploration: float = .1
    reward_scale: float = 50.0
    replay_capacity: int = 100000
    head_steps: int = 100
    rep_steps: int = 100
    repair_steps: int = 100
    adapt_steps: int = 100
    mix: float = .25
    actor_lr: float | None = None
    critic_head_init: str = "random"
    actor_init_fraction: tuple[float, float] | None = None
    critic_action_coordinates: str = "physical"
    actor_warmup_updates: int = 0

    def __post_init__(self):
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Critic learning rate must be finite and positive")
        if self.actor_lr is not None and (not np.isfinite(self.actor_lr) or self.actor_lr <= 0):
            raise ValueError("Actor learning rate must be finite and positive")
        if self.critic_head_init not in {"random", "zero"}:
            raise ValueError("Critic head initialization must be 'random' or 'zero'")
        if self.critic_action_coordinates not in {"physical", "normalized"}:
            raise ValueError("Critic action coordinates must be 'physical' or 'normalized'")
        if (isinstance(self.actor_warmup_updates, (bool, np.bool_))
                or not isinstance(self.actor_warmup_updates, int)
                or self.actor_warmup_updates < 0):
            raise ValueError("actor_warmup_updates must be a nonnegative integer")
        if self.actor_init_fraction is not None and (
                not isinstance(self.actor_init_fraction, tuple)
                or len(self.actor_init_fraction) != 2
                or any(isinstance(fraction, (bool, np.bool_))
                       or not isinstance(fraction, Real)
                       or not math.isfinite(fraction)
                       or not 0 < fraction < 1
                       for fraction in self.actor_init_fraction)):
            raise ValueError("actor_init_fraction must be a tuple of two finite fractions in (0,1)")


class ThermalAdapter:
    """Fixed physical schema, no learned or client-dependent normalization.

    Actor observes nine scalars, previous heater powers (100 MW units), geometry
    and actuator endpoints. By default critic actions are commanded physical
    powers in the same 100 MW units; the explicit normalized-coordinate ablation
    instead uses the actor's [-1, 1] axes. Only the identical TORAX Gaussian deposition family is
    supported; this is not a claim that real NBI and ICRF have identical effects.
    """
    obs_scale = np.array([3, 3, 3, 10, 1, 3, 3, 1, 1], np.float32)
    schema = "thermal-td3-v1:obs9-fixed:prev2-W1e8:BRaIp:bounds4:gaussian-.3-.2-.5,.1-.1"

    def __init__(self, env):
        if env.bank.names != ("aux_heat", "ecrh"):
            raise ValueError("Only the two registered thermal actuators are supported")
        src = env.core.config_dict["sources"]
        for name, fields in {
            "generic_heat": {"gaussian_location": .3, "gaussian_width": .2,
                             "electron_heat_fraction": .5},
            "ecrh": {"gaussian_location": .1, "gaussian_width": .1},
        }.items():
            if any(src[name].get(k) != v for k, v in fields.items()):
                raise ValueError("Unregistered heating deposition semantics")
        self.lo = np.array([s.lo * s.scale / 1e8 for s in env.bank.specs], np.float32)
        self.hi = np.array([s.hi * s.scale / 1e8 for s in env.bank.specs], np.float32)
        d = env.device
        self.context = np.array([d.B_0/10, d.R_major/10, d.a_minor/3,
                                 d.Ip_nominal/2e7, *self.lo, *self.hi], np.float32)

    def observe(self, env, raw):
        if np.shape(raw) != (9,):
            raise ValueError("Observation schema changed")
        previous = env.bank.prev_values * np.array([s.scale for s in env.bank.specs]) / 1e8
        x = np.concatenate([raw / self.obs_scale, previous, self.context]).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observation; do not train on corrupted state")
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, lo, hi, width, action_coordinates="physical"):
        super().__init__()
        if action_coordinates not in {"physical", "normalized"}:
            raise ValueError("Critic action coordinates must be 'physical' or 'normalized'")
        self.action_coordinates = action_coordinates
        self.register_buffer("lo", torch.tensor(lo))
        self.register_buffer("hi", torch.tensor(hi))
        self.trunk = nn.Sequential(nn.Linear(obs_dim + 2, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width, 1)

    def forward(self, obs, action):
        # Saved legacy critics have no coordinate attribute. Preserve their
        # physical-power semantics, including the original operation order.
        if getattr(self, "action_coordinates", "physical") == "physical":
            critic_action = self.lo + (action + 1) * .5 * (self.hi - self.lo)
        else:
            critic_action = action
        return self.head(self.trunk(torch.cat([obs, critic_action], dim=-1)))


def delayed_actor_update_count(start_steps, end_steps, policy_delay, actor_warmup_updates=0):
    """Count actual actor steps in (start_steps, end_steps], after burn-in.

    These counters count critic optimizer calls, not environment transitions.
    Delayed target-network updates during burn-in are not actor optimizer calls.
    """
    values = (start_steps, end_steps, policy_delay, actor_warmup_updates)
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, int)
            for value in values)
            or start_steps < 0 or end_steps < start_steps
            or policy_delay < 1 or actor_warmup_updates < 0):
        raise ValueError("Invalid TD3 update-count interval or schedule")
    return max(0, end_steps // policy_delay - max(start_steps, actor_warmup_updates) // policy_delay)


class TD3:
    def __init__(self, obs_dim, lo, hi, seed, cfg=TD3Config(), representation_seed=1729):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        # Replay draws must not change future exploration or warm-up actions.
        # Keep the original exploration stream and derive a separate replay stream.
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5245504C]))
        self.torch_rng = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.actor = nn.Sequential(nn.Linear(obs_dim, cfg.hidden), nn.ReLU(),
                                       nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(),
                                       nn.Linear(cfg.hidden, 2), nn.Tanh())
            # Start near 12% power, rather than ~50% (an immediate TCV limit event).
            # This initialization is common to scratch and transfer, not transferred.
            nn.init.uniform_(self.actor[-2].weight, -1e-3, 1e-3)
            nn.init.constant_(self.actor[-2].bias, -1.0)
            if cfg.actor_init_fraction is not None:
                # Consume the legacy draws above before applying this explicit
                # calibrated initialization. Hidden layers, critics, and random
                # streams therefore stay identical. Zero final weights make
                # the initial command independent of the observation.
                nn.init.zeros_(self.actor[-2].weight)
                # Equivalent to atanh(2*f-1), without cancellation near zero.
                biases = [0.5 * (math.log(f) - math.log1p(-f))
                          for f in cfg.actor_init_fraction]
                with torch.no_grad():
                    self.actor[-2].bias.copy_(torch.tensor(biases, dtype=self.actor[-2].bias.dtype))
            self.critics = nn.ModuleList()
            for branch in range(2):
                torch.manual_seed(representation_seed + branch)
                c = Critic(obs_dim, lo, hi, cfg.hidden, cfg.critic_action_coordinates)
                # Same trunks per branch across devices; local heads independently seeded.
                torch.manual_seed(seed + 100 + branch)
                c.head.reset_parameters()
                # Consume exactly the established initialization draws before
                # applying the optional zero-value ablation. Trunks are unchanged.
                if cfg.critic_head_init == "zero":
                    nn.init.zeros_(c.head.weight)
                    nn.init.zeros_(c.head.bias)
                self.critics.append(c)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critics = copy.deepcopy(self.critics)
        actor_lr = cfg.lr if cfg.actor_lr is None else cfg.actor_lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr)
        self.replay = []
        self.cursor = self.steps = self.env_steps = 0

    def action(self, obs, explore=False):
        with torch.no_grad():
            a = self.actor(torch.tensor(obs).unsqueeze(0))[0].numpy()
        if explore:
            a = a + self.rng.normal(0, self.cfg.exploration, a.shape)
        return np.clip(a, -1, 1).astype(np.float32)

    def remember(self, obs, action, reward, nxt, terminated):
        row = (obs.copy(), action.copy(), reward / self.cfg.reward_scale,
               nxt.copy(), float(terminated))
        if len(self.replay) < self.cfg.replay_capacity:
            self.replay.append(row)
        else:
            self.replay[self.cursor] = row
        self.cursor = (self.cursor + 1) % self.cfg.replay_capacity
        self.env_steps += 1

    def batch(self):
        ids = self.replay_rng.integers(len(self.replay), size=self.cfg.batch)
        parts = zip(*(self.replay[i] for i in ids))
        o, a, r, n, terminal = [torch.tensor(np.array(x), dtype=torch.float32) for x in parts]
        return o, a, r[:, None], n, terminal[:, None]

    @torch.no_grad()
    def labels(self, r, nxt, terminal):
        noise = torch.randn((len(nxt), 2), generator=self.torch_rng) * self.cfg.target_noise
        a = (self.target_actor(nxt) + noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)).clamp(-1, 1)
        q = torch.minimum(*(c(nxt, a) for c in self.target_critics))
        return r + self.cfg.gamma * (1-terminal) * q

    def train_step(self):
        """Perform one TD3 update and return pre-update batch diagnostics.

        ``None`` means there was insufficient replay and no optimizer ran. Actor
        diagnostics are ``None`` on critic-only steps. These are training-batch
        quantities, not held-out performance or calibrated uncertainty estimates.
        """
        if len(self.replay) < self.cfg.batch:
            return None
        o, a, r, n, terminal = self.batch()
        y = self.labels(r, n, terminal)
        predictions = [c(o, a) for c in self.critics]
        errors = [q-y for q in predictions]
        losses = [error.square().mean() for error in errors]
        loss = sum(losses)
        diagnostics = {
            "critic_loss": float(loss.detach()),
            "q1_loss": float(losses[0].detach()),
            "q2_loss": float(losses[1].detach()),
            "td_abs_mean": float(torch.stack([e.detach().abs().mean() for e in errors]).mean()),
            "q1_mean": float(predictions[0].detach().mean()),
            "q2_mean": float(predictions[1].detach().mean()),
            "target_mean": float(y.mean()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "actor_loss": None,
            "actor_grad_norm": None,
        }
        diagnostics["critic_grad_norm"] = self._optimize(loss, self.critic_opt)
        self.steps += 1
        diagnostics["updates"] = self.steps
        delayed_update = self.steps % self.cfg.policy_delay == 0
        diagnostics["actor_updated"] = delayed_update and self.steps > self.cfg.actor_warmup_updates
        if diagnostics["actor_updated"]:
            for p in self.critics.parameters():
                p.requires_grad_(False)
            try:
                actor_loss = -self.critics[0](o, self.actor(o)).mean()
                diagnostics["actor_loss"] = float(actor_loss.detach())
                diagnostics["actor_grad_norm"] = self._optimize(actor_loss, self.actor_opt)
            finally:
                for p in self.critics.parameters():
                    p.requires_grad_(True)
        # During critic burn-in, keep the established delayed target cadence.
        # Holding the actor fixed must not also freeze the Bellman target critics.
        if delayed_update:
            with torch.no_grad():
                for online, target in [(self.actor, self.target_actor),
                                       (self.critics, self.target_critics)]:
                    for p, t in zip(online.parameters(), target.parameters()):
                        t.lerp_(p, self.cfg.tau)
        return diagnostics

    @staticmethod
    def _optimize(loss, opt):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        opt.zero_grad()
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for g in opt.param_groups for p in g["params"]):
            raise FloatingPointError("Nonfinite gradient")
        grad_norm = float(torch.stack([p.grad.detach().double().square().sum()
                                     for g in opt.param_groups for p in g["params"]
                                     if p.grad is not None]).sum().sqrt())
        opt.step()
        if any(not torch.isfinite(p).all() for g in opt.param_groups for p in g["params"]):
            raise FloatingPointError("Nonfinite parameters")
        return grad_norm

    def representation(self):
        return {f"q{k}.{n}": v.detach().numpy().copy()
                for k, c in enumerate(self.critics) for n, v in c.trunk.state_dict().items()}

    def install(self, values, mix=1.0):
        current = self.representation()
        if set(values) != set(current) or not 0 <= mix <= 1:
            raise ValueError("Invalid representation")
        for n, v in values.items():
            if v.shape != current[n].shape or not np.isfinite(v).all():
                raise ValueError("Invalid representation tensor")
        for k, c in enumerate(self.critics):
            c.trunk.load_state_dict({n: torch.tensor((1-mix)*current[f"q{k}.{n}"] + mix*values[f"q{k}.{n}"])
                                     for n in c.trunk.state_dict()})

    def frozen_dataset(self, batches):
        """Training replay only; draw batches and labels ONCE before fitting."""
        data = []
        for _ in range(batches):
            o, a, r, n, terminal = self.batch()
            data.append((o, a, self.labels(r, n, terminal)))
        scale = max(float(torch.cat([d[2] for d in data]).square().mean().sqrt()), 1.0)
        return data, scale

    def fit_parts(self, data, scale, part, count, lr):
        for c in self.critics:
            for p in c.parameters():
                p.requires_grad_(False)
            for p in getattr(c, part).parameters():
                p.requires_grad_(True)
        opt = torch.optim.SGD([p for c in self.critics for p in getattr(c, part).parameters()], lr=lr)
        try:
            for i in range(count):
                o, a, y = data[i % len(data)]
                loss = sum(((c(o, a)-y)/scale).square().mean() for c in self.critics)
                self._optimize(loss, opt)
        finally:
            for p in self.critics.parameters():
                p.requires_grad_(True)

    def contribution(self, base, source, update_id, schema, revision):
        if len(self.replay) < self.cfg.batch:
            raise ValueError("Insufficient training replay")
        job = copy.deepcopy(self)
        job.install(base)
        data, scale = job.frozen_dataset(max(self.cfg.head_steps, self.cfg.rep_steps))
        job.fit_parts(data, scale, "head", self.cfg.head_steps, self.cfg.rep_lr)
        job.fit_parts(data, scale, "trunk", self.cfg.rep_steps, self.cfg.rep_lr)
        delta = {k: v-base[k] for k, v in job.representation().items()}
        return RepresentationDelta(update_id, source, model_hash(base), schema, revision, delta)

    def repaired_fork(self, proposal=None):
        actor_updates = delayed_actor_update_count(
            self.steps, self.steps + self.cfg.adapt_steps,
            self.cfg.policy_delay, self.cfg.actor_warmup_updates)
        if (len(self.replay) < self.cfg.batch or self.cfg.adapt_steps < self.cfg.policy_delay
                or actor_updates == 0):
            raise ValueError("Transfer gate requires enough replay and actual actor learning")
        job = copy.deepcopy(self)
        if proposal is not None:
            job.install(proposal, self.cfg.mix)
        data, scale = job.frozen_dataset(self.cfg.repair_steps)
        job.fit_parts(data, scale, "head", self.cfg.repair_steps, self.cfg.rep_lr)
        job.target_critics = copy.deepcopy(job.critics)
        job.critic_opt = torch.optim.Adam(job.critics.parameters(), lr=self.cfg.lr)
        for _ in range(self.cfg.adapt_steps):
            job.train_step()
        return job


def channel_hash(cfg):
    return hashlib.sha256(json.dumps({"adapter": ThermalAdapter.schema,
                                      "td3": asdict(cfg)}, sort_keys=True).encode()).hexdigest()


def acceptance(control, candidate, control_events, candidate_events, scale, seed,
               min_episodes=32, bootstrap=10000, tolerance=.05):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Gate tolerance must be finite and nonnegative")
    a, b = np.asarray(control), np.asarray(candidate)
    if (len(a) < min_episodes or a.shape != b.shape or scale <= 0
            or not np.isfinite(scale) or not np.isfinite(a).all() or not np.isfinite(b).all()):
        return {"accept": False, "reason": "insufficient_or_nonfinite"}
    if len(control_events) != len(a) or len(candidate_events) != len(a):
        return {"accept": False, "reason": "event_length_mismatch"}
    new_event = any(set(y) - set(x) for x, y in zip(control_events, candidate_events))
    d = (b-a)/scale
    rng = np.random.default_rng(seed)
    upper = float(np.quantile(d[rng.integers(len(d), size=(bootstrap, len(d)))].mean(1), .95))
    return {"accept": bool(upper <= tolerance and not new_event), "upper95": upper,
            "mean_difference": float(d.mean()), "new_hard_event": bool(new_event)}
