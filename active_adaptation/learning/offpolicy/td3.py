# bash scripts/launch_ddp.sh \
#   0,1,2,3,4,5,6,7 \
#   scripts/train_ppo.py \
#   venv/mjlab \
#   task=lafan \
#   algo=td3_offpolicy \
#   algo.normalize_reward=True \
#   algo.grad_sync_mode=manual \
#   backend=mjlab \
#   total_frames=655360000 \
#   task.num_envs=2048 \
#   task.reward.loco.feet_air_time.weight=8.0


import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictModuleBase,
    TensorDictSequential as Seq,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import Composite, TensorSpec
from torchrl.objectives import hold_out_net

import active_adaptation as aa
from active_adaptation.learning.modules import VecNorm, IndependentNormal
from active_adaptation.learning.offpolicy.buffer import ReplayBuffer
from active_adaptation.learning.offpolicy.distributional import (
    ValueDistribution,
    expected_q_from_logits,
)
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.offpolicy.sac import TwinDistributionalQNetwork
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    OBS_KEY,
    OBS_PRIV_KEY,
    REWARD_KEY,
    TERM_KEY,
    CatTensors,
    soft_copy_,
)

cs = ConfigStore.instance()

ACTOR_INPUT_KEY = "_actor_input"
CRITIC_INPUT_KEY = "_critic_input"
Q_OUTPUT_KEY = "_q_output"


def _init_td3_linear(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


def _normalize_grad_sync_mode(mode: str | None) -> str | None:
    if isinstance(mode, str):
        mode = mode.lower()
        if mode in {"none", "null"}:
            mode = None
    if mode not in {"manual", "ddp", None}:
        raise ValueError(
            "grad_sync_mode must be one of {'manual', 'ddp', None}, "
            f"got {mode!r}"
        )
    return mode


class DDPWithAttr(DDP):
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is not None and hasattr(module, name):
                return getattr(module, name)
            raise


class TwinQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        critic_input_dim = obs_dim + act_dim
        self.critic_1 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1),
        )
        self.critic_2 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Linear(hidden_dims[2], 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.critic_1.apply(_init_td3_linear)
        self.critic_2.apply(_init_td3_linear)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        q1 = self.critic_1(x)
        q2 = self.critic_2(x)
        return torch.cat([q1, q2], dim=-1)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, ...] = (384, 384, 384),
        action_init: Literal["zeros", "orthogonal"] = "zeros",
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dims[2], act_dim)
        self.fc_logstd = nn.Linear(hidden_dims[2], act_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.trunk.apply(_init_td3_linear)
        if action_init == "orthogonal":
            self.fc_mu.apply(lambda m: _init_td3_linear(m, gain=0.01))
            self.fc_logstd.apply(lambda m: _init_td3_linear(m, gain=0.01))
        elif action_init == "zeros":
            nn.init.constant_(self.fc_mu.weight, 0.0)
            nn.init.constant_(self.fc_mu.bias, 0.0)
            nn.init.constant_(self.fc_logstd.weight, 0.0)
            nn.init.constant_(self.fc_logstd.bias, 0.0)
        else:
            raise ValueError(f"Invalid action_init: {action_init}")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(obs)
        loc = self.fc_mu(feat)
        raw = self.fc_logstd(feat)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1.0 + torch.tanh(raw)
        )
        scale = torch.exp(log_std)
        return loc, scale


@dataclass
class TD3Config:
    _target_: str = "active_adaptation.learning.offpolicy.td3.TD3"
    name: str = "td3_offpolicy"

    train_every: int = 4
    buffer_size: int = 2000
    buffer_device: str = "cpu"
    warm_up_steps: int = 200

    lr_actor: float = 5e-4
    lr_critic: float = 5e-4
    weight_decay: float = 0.02
    # Q 网络对 L2 衰减很敏感，默认单独关掉 critic 的 weight decay 更稳。
    critic_weight_decay: float = 0.0

    n_steps: int = 3
    gamma: float = 0.95
    utd_ratio: int = 4
    policy_frequency: int = 4

    actor_hidden_dims: Tuple[int, ...] = (384, 384, 384)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    actor_init: str = "zeros"
    log_std_min: float = -5.0
    log_std_max: float = 0.0

    critic_batch_size: int = 4096
    actor_batch_size: int = 4096

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.2
    tau_actor: float = 0.1
    tau_q: float = 0.01
    max_grad_norm: float = 1.0
    critic_max_grad_norm: float = 1
    # Bellman 残差用 Huber（smooth L1）比纯 MSE 更不炸；设为 None 则退回 MSE。
    critic_huber_beta: float | None = 1.0
    # Penalize large pre-tanh actor means (same form as SAC).
    actor_loc_reg_weight: float = 0.01
    actor_loc_reg_scale: float = 8.0
    actor_loc_reg_power: float = 6.0
    # C51-style twin critic (same spirit as :class:`SAC`); TD3 target has no entropy in reward.
    distributional: bool = True
    # Debug: set reward to (1-gamma) and verify q_value converges to ~1.
    debug: bool = False
    # FlashSAC-style running return scaling (see :class:`RewardNormalizer`). Independent of q_support.
    normalize_reward: bool = True
    # Only for :class:`RewardNormalizer` denominator; C51 grid uses q_support_* below.
    normalized_G_max: float = 6.0
    reward_norm_epsilon: float = 1e-8
    # Distributional critic support on the real line (C51 atoms).
    q_support_min: float = -5.0
    q_support_max: float = 5.0
    q_support_atom_delta: float = 0.05

    vecnorm: bool = True
    # "manual" all-reduces grads (no DDP hooks). DDP on the actor breaks easily because
    # TD3 policy loss only flows through loc while GaussianActor also computes scale.
    grad_sync_mode: str | None = "manual"

    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)

    def __post_init__(self):
        self.utd_ratio = int(self.utd_ratio)
        if self.utd_ratio < 1:
            raise ValueError(f"utd_ratio must be >= 1, got {self.utd_ratio}.")
        self.policy_frequency = int(self.policy_frequency)
        if self.policy_frequency < 1:
            raise ValueError(
                f"policy_frequency must be >= 1, got {self.policy_frequency}."
            )
        self.actor_hidden_dims = tuple(int(x) for x in self.actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(x) for x in self.critic_hidden_dims)
        if len(self.actor_hidden_dims) != 3:
            raise ValueError("actor_hidden_dims must contain 3 dims.")
        if len(self.critic_hidden_dims) != 3:
            raise ValueError("critic_hidden_dims must contain 3 dims.")
        if self.actor_loc_reg_weight < 0:
            raise ValueError(
                f"actor_loc_reg_weight must be >= 0, got {self.actor_loc_reg_weight}."
            )
        if self.actor_loc_reg_scale <= 0:
            raise ValueError(
                f"actor_loc_reg_scale must be > 0, got {self.actor_loc_reg_scale}."
            )
        if self.actor_loc_reg_power <= 0:
            raise ValueError(
                f"actor_loc_reg_power must be > 0, got {self.actor_loc_reg_power}."
            )
        if bool(getattr(self, "distributional", False)):
            w = self.critic_hidden_dims[0]
            if any(int(d) != w for d in self.critic_hidden_dims):
                raise ValueError(
                    "distributional TD3 requires critic_hidden_dims all equal "
                    f"(TwinDistributionalQNetwork), got {self.critic_hidden_dims}."
                )


cs.store(name="td3_offpolicy", node=TD3Config, group="algo")


class TD3(TensorDictModuleBase):
    def __init__(
        self,
        cfg: TD3Config,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.env = env

        self.policy_frequency = int(getattr(self.cfg, "policy_frequency", 2))
        self.grad_sync_mode = _normalize_grad_sync_mode(
            getattr(self.cfg, "grad_sync_mode", "manual")
        )
        self.world_size = aa.get_world_size()
        self._distributed = aa.is_distributed()
        if self._distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Distributed training is enabled but torch.distributed is not initialized."
            )

        required_obs_keys = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)
        observation_keys = set(observation_spec.keys(True, True))
        missing_keys = sorted(set(required_obs_keys).difference(observation_keys))
        if missing_keys:
            raise KeyError(
                f"TD3 requires observation keys {required_obs_keys}, missing {missing_keys}."
            )

        self.actor_obs_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY)
        self.critic_obs_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)
        self.actor_obs_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.actor_obs_keys
        )
        self.critic_obs_dim = sum(
            int(observation_spec[key].shape[-1]) for key in self.critic_obs_keys
        )
        self.act_dim = int(action_spec.shape[-1])
        _dev = torch.device(device) if not isinstance(device, torch.device) else device

        vecnorm_modules = []
        self.vecnorms = nn.ModuleDict()
        for key in self.critic_obs_keys:
            shape = observation_spec[key].shape[-1:]
            vecnorm = (
                VecNorm(input_shape=shape, stats_shape=shape, decay=0.999)
                if self.cfg.vecnorm
                else nn.Identity()
            )
            self.vecnorms[key] = vecnorm
            vecnorm_modules.append(Mod(vecnorm, [key], [key]))
        self.vecnorm = Seq(*vecnorm_modules).to(device)

        actor_net = GaussianActor(
            self.actor_obs_dim,
            self.act_dim,
            hidden_dims=self.cfg.actor_hidden_dims,
            action_init=self.cfg.actor_init,
            log_std_min=self.cfg.log_std_min,
            log_std_max=self.cfg.log_std_max,
        )
        if bool(getattr(self.cfg, "distributional", False)):
            v_min = float(getattr(self.cfg, "q_support_min", -1.0))
            v_max = float(getattr(self.cfg, "q_support_max", 9.0))
            delta = float(getattr(self.cfg, "q_support_atom_delta", 0.05))
            if v_max <= v_min or delta <= 0:
                raise ValueError(
                    f"Need q_support_max > q_support_min and q_support_atom_delta > 0; "
                    f"got min={v_min}, max={v_max}, delta={delta}."
                )
            num_atoms = int((v_max - v_min) / delta) + 1
            if num_atoms < 3:
                raise ValueError(
                    f"distributional TD3 needs num_atoms >= 3, got {num_atoms} "
                    f"(min={v_min}, max={v_max}, delta={delta})."
                )
            self.register_buffer(
                "q_support",
                torch.linspace(v_min, v_max, num_atoms, device=_dev),
            )
            q_net = TwinDistributionalQNetwork(
                self.critic_obs_dim,
                self.act_dim,
                num_atoms=num_atoms,
                hidden_dims=self.cfg.critic_hidden_dims,
                simba_mlp=False,
            )
        else:
            q_net = TwinQNetwork(
                self.critic_obs_dim,
                self.act_dim,
                hidden_dims=self.cfg.critic_hidden_dims,
            )

        self.actor = Seq(
            CatTensors(
                self.actor_obs_keys,
                ACTOR_INPUT_KEY,
                del_keys=False,
                sort=False,
            ),
            Mod(actor_net, [ACTOR_INPUT_KEY], ["loc", "scale"]),
        ).to(device)
        self.Q = Seq(
            CatTensors(
                self.critic_obs_keys,
                CRITIC_INPUT_KEY,
                del_keys=False,
                sort=False,
            ),
            Mod(q_net, [CRITIC_INPUT_KEY, ACTION_KEY], [Q_OUTPUT_KEY]),
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target.requires_grad_(False)
        self.Q_target.requires_grad_(False)

        if self._distributed:
            if self.grad_sync_mode == "ddp":
                self._wrap_ddp(local_rank=aa.get_local_rank())
            self._broadcast_parameters()

        self.opt_actor = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.cfg.lr_actor,
            weight_decay=self.cfg.weight_decay,
        )
        self.opt_Q = torch.optim.AdamW(
            self.Q.parameters(),
            lr=self.cfg.lr_critic,
            weight_decay=float(
                getattr(self.cfg, "critic_weight_decay", self.cfg.weight_decay)
            ),
        )

        self.global_step = 0
        self.gradient_step = 0
        self.enable_actor = True

        if env is None:
            raise ValueError("TD3 requires env for ReplayBuffer layout (fake_tensordict).")
        if cfg.buffer_device == "cuda":
            cfg.buffer_device = device
        next_obs_keys = tuple(("next", key) for key in self.critic_obs_keys)
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), *next_obs_keys, "collector")
            .detach()
        ).to(cfg.buffer_device)
        fake_rb[REWARD_KEY] = fake_rb[REWARD_KEY].sum(-1, keepdim=True)
        self.rb = ReplayBuffer(
            self.cfg.buffer_size,
            fake_rb,
            gamma=self.cfg.gamma,
            obs_keys=self.critic_obs_keys,
        )

        self.reward_normalizer: RewardNormalizer | None = None
        if bool(getattr(self.cfg, "normalize_reward", False)):
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                G_max=float(self.cfg.normalized_G_max),
                load_rms=False,
                device=_dev,
                epsilon=float(getattr(self.cfg, "reward_norm_epsilon", 1e-8)),
            )

    def _unwrap_module(self, module: nn.Module) -> nn.Module:
        return module.module if isinstance(module, DDP) else module

    def _wrap_ddp(self, local_rank: int) -> None:
        device = (
            torch.device(self.device)
            if not isinstance(self.device, torch.device)
            else self.device
        )
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": True,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)

        self.actor = DDPWithAttr(self.actor, **ddp_kwargs)
        self.Q = DDPWithAttr(self.Q, **ddp_kwargs)

    @torch.no_grad()
    def _broadcast_parameters(self) -> None:
        if not self._distributed:
            return
        for module in (
            self.vecnorm,
            self.actor,
            self.actor_target,
            self.Q,
            self.Q_target,
        ):
            for param in module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    @torch.no_grad()
    def _broadcast_buffers(self, *modules: nn.Module) -> None:
        if not self._distributed:
            return
        for module in modules:
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    @torch.no_grad()
    def _all_reduce_grads(self, *modules: nn.Module) -> None:
        if not self._distributed:
            return
        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def _sync_vecnorms(self) -> None:
        if not self._distributed or not self.cfg.vecnorm:
            return
        for vecnorm in self.vecnorms.values():
            if hasattr(vecnorm, "synchronize"):
                vecnorm.synchronize(mode="broadcast")
            else:
                self._broadcast_buffers(vecnorm)

    def on_stage_start(self, stage: str):
        del stage
        self.enable_actor = True

    def _actor_dist(
        self,
        tensordict: TensorDict,
        actor_module: nn.Module | None = None,
    ) -> IndependentNormal:
        if actor_module is None:
            actor_module = self.actor
        actor_module(tensordict)
        return IndependentNormal(tensordict["loc"], tensordict["scale"])

    def get_rollout_policy(self, mode: str = "train", critic: bool = False):
        del critic

        @torch.no_grad()
        def policy(tensordict: TensorDict):
            work_td = tensordict.copy()
            with VecNorm.freeze():
                self.vecnorm(work_td)
            dist = self._actor_dist(work_td)
            action = dist.sample()
            if mode == "train" and self.cfg.exploration_noise > 0:
                action = action + torch.randn_like(action) * self.cfg.exploration_noise
            tensordict[ACTION_KEY] = action
            tensordict["loc"] = work_td["loc"]
            tensordict["scale"] = work_td["scale"]
            return tensordict

        return policy

    def train_op(self, tensordict: TensorDict):
        self.global_step += self.cfg.train_every

        next_obs_keys = tuple(("next", key) for key in self.critic_obs_keys)
        td = tensordict.exclude(("next", "stats"), *next_obs_keys, "collector")
        td = td.copy()
        if self.cfg.debug:
            # Debug: constant reward scaled by effective horizon.
            # Value target should converge to ~1 in this case.
            reward = td[REWARD_KEY].sum(-1, keepdim=True)
            td[REWARD_KEY] = torch.ones_like(reward) * (1.0 - self.cfg.gamma)
            neg_rew_ratio = 0.0
        else:
            # Match :class:`SAC` reward pipeline before replay (non-negative shaped rewards for RN / critic).
            reward = td[REWARD_KEY].sum(-1, keepdim=True)
            neg_rew_ratio = (reward <= 0.0).float().mean().item()
            td[REWARD_KEY] = reward.clamp_min(0.0)

        bs = td.batch_size
        if len(bs) >= 2:
            for ti in range(int(bs[1])):
                sub = td[:, ti]
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update_reward_stats(
                        reward=sub[REWARD_KEY],
                        terminated=sub[TERM_KEY],
                        truncated=sub["next", "truncated"],
                    )
                self.rb.push(sub)
        else:
            if self.reward_normalizer is not None:
                self.reward_normalizer.update_reward_stats(
                    reward=td[REWARD_KEY],
                    terminated=td[TERM_KEY],
                    truncated=td["next", "truncated"],
                )
            self.rb.push(td)

        infos: dict[str, float] = {
            "rb_size": float(len(self.rb)),
            "critic/neg_rew_ratio": neg_rew_ratio,
        }
        if self.global_step < self.cfg.warm_up_steps:
            self._sync_vecnorms()
            return infos

        iters = self.cfg.train_every * self.cfg.utd_ratio
        critic_batch = self.rb.sample(
            batch_size=self.cfg.critic_batch_size * iters,
            steps=self.cfg.n_steps,
        ).to(self.device)
        self.vecnorm(critic_batch)
        self.vecnorm(critic_batch["next"])

        actor_update_count = 0
        if self.enable_actor:
            actor_update_count = sum(
                1
                for j in range(iters)
                if (self.gradient_step + j) % self.policy_frequency == 0
            )
        actor_batch = None
        if actor_update_count:
            actor_batch = self.rb.sample(
                batch_size=self.cfg.actor_batch_size * actor_update_count,
                steps=1,
            ).to(self.device)
            self.vecnorm(actor_batch)
            self.vecnorm(actor_batch["next"])

        actor_update_idx = 0
        critic_info: dict[str, float] = {}
        actor_info: dict[str, float] = {}
        for i in range(iters):
            s = i * self.cfg.critic_batch_size
            e = s + self.cfg.critic_batch_size
            batch = critic_batch[s:e]
            critic_info = self.train_critic(batch, diagnostics=(i == iters - 1))

            if (
                self.enable_actor
                and actor_batch is not None
                and self.gradient_step % self.policy_frequency == 0
            ):
                s = actor_update_idx * self.cfg.actor_batch_size
                e = s + self.cfg.actor_batch_size
                actor_info = self.train_actor(
                    batch=actor_batch[s:e],
                    diagnostics=(actor_update_idx == actor_update_count - 1),
                )
                actor_update_idx += 1
            self.gradient_step += 1

        infos.update(critic_info)
        infos.update(actor_info)
        self._sync_vecnorms()
        return dict(sorted(infos.items()))

    def train_critic(self, batch: TensorDict, diagnostics: bool = False):
        self.Q.train()
        batch = batch.copy()
        reward = batch[REWARD_KEY]
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        next_batch = batch["next"].copy()
        discount = next_batch["discount"]

        with torch.no_grad():
            # TD3 target policy smoothing: a' = μ(s') + clip(ε), ε ~ N(0, σ).
            # Do not use dist.sample() — that adds learned σ on top of smoothing and
            # can destabilize Q targets when scale is large.
            self._actor_dist(next_batch, self.actor_target)
            target_noise = (
                torch.randn_like(next_batch["loc"]) * self.cfg.policy_noise
            )
            target_noise = target_noise.clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_action = next_batch["loc"] + target_noise
            next_batch[ACTION_KEY] = next_action

            next_logits = self.Q_target(next_batch)[Q_OUTPUT_KEY]
            if self.cfg.distributional:
                n1, n2 = next_logits.chunk(2, dim=-1)
                p1 = ValueDistribution(n1, self.q_support).project(reward, discount)
                p2 = ValueDistribution(n2, self.q_support).project(reward, discount)
                z = self.q_support.to(device=p1.device, dtype=p1.dtype).view(1, -1)
                ev1 = (p1 * z).sum(-1, keepdim=True)
                ev2 = (p2 * z).sum(-1, keepdim=True)
                q_target = torch.where(ev1 < ev2, p1, p2)
            else:
                next_q = next_logits.min(dim=-1, keepdim=True).values
                q_target = reward + discount * next_q

        qs = self.Q(batch)[Q_OUTPUT_KEY]
        if self.cfg.distributional:
            q1, q2 = qs.chunk(2, dim=-1)
            log_p1 = F.log_softmax(q1, dim=-1).clamp(min=-30.0)
            log_p2 = F.log_softmax(q2, dim=-1).clamp(min=-30.0)
            q_loss = -((q_target * log_p1).sum(-1) + (q_target * log_p2).sum(-1)).mean()
        else:
            q_tgt = q_target.expand_as(qs)
            beta = getattr(self.cfg, "critic_huber_beta", None)
            if beta is not None and beta > 0:
                q_loss = F.smooth_l1_loss(qs, q_tgt, beta=float(beta))
            else:
                q_loss = F.mse_loss(qs, q_tgt)

        self.opt_Q.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.grad_sync_mode == "manual":
            self._all_reduce_grads(self.Q)
        q_clip = float(
            getattr(self.cfg, "critic_max_grad_norm", self.cfg.max_grad_norm)
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.Q.parameters(),
            max_norm=q_clip,
        )
        self.opt_Q.step()
        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_q)

        infos = {"critic/q_loss": q_loss.item()}
        if diagnostics:
            with torch.no_grad():
                if self.cfg.distributional:
                    q1, q2 = qs.chunk(2, dim=-1)
                    e1 = expected_q_from_logits(q1, self.q_support)
                    e2 = expected_q_from_logits(q2, self.q_support)
                    q_pair = torch.cat([e1, e2], dim=-1)
                    q_min = q_pair.min(dim=-1).values
                else:
                    q_min = qs.min(dim=-1).values
            infos.update(
                {
                    "critic/q_value": q_min.mean().item(),
                    "critic/q_max": q_min.max().item(),
                    "critic/q_std": q_min.std().item(),
                    "critic/grad_norm": critic_grad_norm.item(),
                }
            )
        return infos

    def train_actor(self, batch: TensorDict, diagnostics: bool = False):
        batch = batch.copy()
        with hold_out_net(self.Q):
            self.actor(batch)
            batch[ACTION_KEY] = batch["loc"]
            qs = self.Q(batch)[Q_OUTPUT_KEY]
            if self.cfg.distributional:
                q1, q2 = qs.chunk(2, dim=-1)
                e1 = expected_q_from_logits(q1, self.q_support)
                e2 = expected_q_from_logits(q2, self.q_support)
                q_scalar = torch.cat([e1, e2], dim=-1)
                q_value = q_scalar.min(dim=-1).values
            else:
                q_value = qs.min(dim=-1).values
            policy_term = -q_value
            loc = batch["loc"]
            loc_reg = (
                self.cfg.actor_loc_reg_weight
                * (loc.abs() / self.cfg.actor_loc_reg_scale)
                .pow(self.cfg.actor_loc_reg_power)
                .sum(-1)
                .reshape_as(policy_term)
            )
            actor_loss = (policy_term + loc_reg).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_sync_mode == "manual":
            self._all_reduce_grads(self.actor)
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=self.cfg.max_grad_norm,
        )
        self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        if not diagnostics:
            return {}
        return {
            "actor/loss": actor_loss.item(),
            "actor/grad_norm": actor_grad_norm.item(),
            "actor/q_mean": q_value.mean().item(),
            "actor/action_abs": batch[ACTION_KEY].abs().mean().item(),
        }

    def state_dict(self):
        state_dict = OrderedDict()
        actor = self._unwrap_module(self.actor)
        Q = self._unwrap_module(self.Q)
        state_dict["actor"] = actor.state_dict()
        state_dict["Q"] = Q.state_dict()
        state_dict["vecnorm"] = self.vecnorm.state_dict()
        state_dict["opt_actor"] = self.opt_actor.state_dict()
        state_dict["opt_Q"] = self.opt_Q.state_dict()
        state_dict["global_step"] = self.global_step
        state_dict["gradient_step"] = self.gradient_step
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        actor = self._unwrap_module(self.actor)
        Q = self._unwrap_module(self.Q)
        actor.load_state_dict(state_dict["actor"], strict=strict)
        Q.load_state_dict(state_dict["Q"], strict=strict)
        self.actor_target.load_state_dict(state_dict["actor"], strict=strict)
        self.Q_target.load_state_dict(state_dict["Q"], strict=strict)
        self.vecnorm.load_state_dict(state_dict["vecnorm"], strict=strict)
        if "opt_actor" in state_dict:
            self.opt_actor.load_state_dict(state_dict["opt_actor"])
        if "opt_Q" in state_dict:
            self.opt_Q.load_state_dict(state_dict["opt_Q"])
        self.global_step = int(state_dict.get("global_step", 0))
        self.gradient_step = int(state_dict.get("gradient_step", 0))
