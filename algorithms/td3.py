import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import torchkit.pytorch_utils as ptu


class TD3(nn.Module):
    def __init__(
        self,
        policy,
        q1_network,
        q2_network,

        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=5e-3,

        max_action=1.0,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        clip_grad_value=None,
    ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.clip_grad_value = clip_grad_value

        # critics
        self.qf1 = q1_network
        self.qf2 = q2_network
        self.qf1_optim = Adam(self.qf1.parameters(), lr=critic_lr)
        self.qf2_optim = Adam(self.qf2.parameters(), lr=critic_lr)

        # target critics
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        # actor
        self.policy = policy
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

        # target actor
        self.policy_target = copy.deepcopy(self.policy)

        # internal step counter for delayed policy update
        self.total_update_steps = 0

    def forward(self, obs):
        action = self.policy(obs)
        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        return action, q1, q2

    def act(self, obs, deterministic=True, return_log_prob=False):
        """
        For TD3, actor is deterministic.
        Keep this function name to align with SAC-style interface.
        """
        action, mean, log_std, log_prob = self.policy(obs, 
                                                      deterministic=True,
                                                      return_log_prob=return_log_prob)
        return action, mean, log_std, log_prob

    def select_action(self, state):
        state = torch.FloatTensor(state).to(ptu.device).unsqueeze(0)
        action, _, _, _ = self.act(state)
        return action.detach().cpu().numpy()[0]

    def _min_q(self, obs, action):
        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        return torch.min(q1, q2)

    def _get_target_action(self, next_obs):
        """
        TD3 target policy smoothing:
        a' = clip( pi_target(s') + clip(eps, -c, c), action_low, action_high )
        This assumes the policy has:
            - action_dim
            - max_action
        and outputs already-scaled actions in [-max_action, max_action].
        """
        next_action, _, _, _ = self.policy(next_obs, 
                                            deterministic=True,
                                            return_log_prob=False)

        noise = torch.randn_like(next_action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = next_action + noise
        next_action = next_action.clamp(-self.max_action, self.max_action)
        return next_action

    def update(self, obs, action, reward, next_obs, done, **kwargs):
        """
        One TD3 update step.
        Inputs are tensors already on the correct device:
            obs:      [B, obs_dim]
            action:   [B, act_dim]
            reward:   [B, 1]
            next_obs: [B, obs_dim]
            done:     [B, 1]
        """
        self.total_update_steps += 1

        # -------- critic update --------
        with torch.no_grad():
            next_action = self._get_target_action(next_obs)
            next_q1 = self.qf1_target(next_obs, next_action)
            next_q2 = self.qf2_target(next_obs, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            q_target = reward + (1.0 - done) * self.gamma * min_next_q

        q1_pred = self.qf1(obs, action)
        q2_pred = self.qf2(obs, action)

        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf1)
        self.qf1_optim.step()

        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf2)
        self.qf2_optim.step()

        # -------- delayed actor update --------
        policy_loss = torch.tensor(0.0, device=ptu.device)

        if self.total_update_steps % self.policy_delay == 0:
            # freeze critics for efficiency
            self._set_requires_grad(self.qf1, False)
            self._set_requires_grad(self.qf2, False)

            new_action, _, _, _ = self.act(obs)
            policy_loss = -self.qf1(obs, new_action).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            if self.clip_grad_value is not None:
                self._clip_grads(self.policy)
            self.policy_optim.step()

            self._set_requires_grad(self.qf1, True)
            self._set_requires_grad(self.qf2, True)

            # soft update targets only when actor updates
            self.soft_target_update()

        return {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_entropy_loss': 0.0, 
            'q_target': torch.mean(q_target).item(),
            'qf1_pred': torch.mean(q1_pred).item(), 
            'qf2_pred': torch.mean(q2_pred).item(), 
            'log_prob': 0.0
        }

    def update_critic(self, obs, action, reward, next_obs, done, **kwargs):
        with torch.no_grad():
            next_action = self._get_target_action(next_obs)
            next_q1 = self.qf1_target(next_obs, next_action)
            next_q2 = self.qf2_target(next_obs, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            q_target = reward + (1.0 - done) * self.gamma * min_next_q

        q1_pred = self.qf1(obs, action)
        q2_pred = self.qf2(obs, action)

        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf1)
        self.qf1_optim.step()

        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf2)
        self.qf2_optim.step()

        return {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
        }

    def update_actor(self, obs, action=None, reward=None, next_obs=None, done=None, **kwargs):
        self._set_requires_grad(self.qf1, False)
        self._set_requires_grad(self.qf2, False)

        new_action, _, _, _ = self.act(obs)
        policy_loss = -self.qf1(obs, new_action).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.policy)
        self.policy_optim.step()

        self._set_requires_grad(self.qf1, True)
        self._set_requires_grad(self.qf2, True)

        self.soft_target_update()

        return {
            'policy_loss': policy_loss.item(),
        }

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def _clip_grads(self, net):
        for p in net.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-self.clip_grad_value, self.clip_grad_value)

    def _set_requires_grad(self, net, requires_grad):
        for p in net.parameters():
            p.requires_grad = requires_grad