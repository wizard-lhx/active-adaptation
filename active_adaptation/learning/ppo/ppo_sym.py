"""PPO with mirrored data augmentation and an explicit policy symmetry loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as distr
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from active_adaptation.learning.modules import IndependentNormal
from active_adaptation.learning.ppo.common import ACTION_KEY
from active_adaptation.learning.ppo.ppo_symaug_eff import PPOConfig as BaseConfig
from active_adaptation.learning.ppo.ppo_symaug_eff import PPOPolicy as BasePolicy
from active_adaptation.utils.profiling import ScopedTimer


@dataclass
class PPOConfig(BaseConfig):
    _target_: str = "active_adaptation.learning.ppo.ppo_sym.PPOConfig"
    name: str = "ppo_sym"
    symmetry_coef: float = 0.1

    def get_class(self):
        return PPOPolicy


ConfigStore.instance().store("ppo_sym", node=PPOConfig, group="algo")


class PPOPolicy(BasePolicy):
    """Add mean-action equivariance loss to the ppo_symaug_eff update."""

    @ScopedTimer("ppo_update")
    def _update(self, tensordict: TensorDict, compute_diagnostics: bool = False):
        bsize = tensordict.shape[0] // 2

        self.vecnorm(tensordict)

        valid = (~tensordict["is_init"]).float()
        valid_cnt = valid.sum()
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        ret = tensordict["ret"]
        log_ratio = (log_probs - log_probs_data).reshape_as(adv)
        ratio = torch.exp(log_ratio)
        eps_neg, eps_pos = self.clip_param
        ratio_det = ratio.detach()
        clamped_pos = ratio_det > 1.0 + eps_pos
        clamped_neg = ratio_det < 1.0 - eps_neg
        clamped = (clamped_pos | clamped_neg).reshape_as(ret)

        policy_loss = self.actor_loss_fn(ratio, adv, self.clip_param)
        entropy_loss = -self.entropy_coef * entropy

        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(ret, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        mean_original = dist.mean[:bsize]
        mean_mirrored = dist.mean[bsize:]
        symmetry_error = (
            mean_mirrored - self.act_transform(mean_original)
        ).square().mean(-1, keepdim=True)
        valid_original = valid[:bsize]
        symmetry_loss = (
            symmetry_error * valid_original
        ).sum() / valid_original.sum().clamp_min(1.0)
        weighted_symmetry_loss = self.cfg.symmetry_coef * symmetry_loss

        loss = policy_loss + entropy_loss + value_loss + weighted_symmetry_loss
        if self.cfg.aux_coef > 0.0:
            aux_weight = clamped.float() * valid
            aux_loss = (
                tensordict["aux_pred"].reshape_as(ret) - ret
            ).square() * aux_weight
            aux_loss = aux_loss.sum() / aux_weight.sum().clamp_min(1.0)
            loss += (
                self.cfg.aux_coef
                * aux_loss
                / max(self.ret_std_ema, 1.0) ** 2
            )
        else:
            aux_loss = ret.new_zeros(())

        self.opt.zero_grad()
        loss.backward()

        if self.should_reduce_grads:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.opt.step()

        if not compute_diagnostics:
            return

        with torch.no_grad():
            explained_var = 1 - F.mse_loss(values, ret) / ret.var()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            actor_feature_norm = torch.norm(
                tensordict["_actor_feature"], dim=-1
            ).mean()
            critic_feature_norm = torch.norm(
                tensordict["_critic_feature"], dim=-1
            ).mean()

        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_pos": clamped_pos.float().mean(),
            "actor/clamp_neg": clamped_neg.float().mean(),
            "actor/approx_kl": approx_kl,
            "actor/aux_loss": aux_loss.detach(),
            "actor/symmetry_loss": symmetry_loss.detach(),
            "actor/symmetry_weighted_loss": weighted_symmetry_loss.detach(),
            "actor/symmetry_coef": symmetry_loss.new_tensor(
                self.cfg.symmetry_coef
            ),
            "actor/feature_norm": actor_feature_norm.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
            "critic/feature_norm": critic_feature_norm.detach(),
        }
