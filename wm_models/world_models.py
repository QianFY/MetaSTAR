import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

from wm_models.functions_losses import SymLogTwoHotLoss
from wm_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from wm_models.transformer_model import StochasticTransformerKVCache, StochasticTransformer

import torchkit.pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from algorithms.focalsac import FOCALSAC
from utils import helpers as utl
import mbrl.models as models
import torch.distributions as dist


class DistHead(nn.Module):
    '''
    Gaussian distribution head for VAE
    '''
    def __init__(self, state_feat_dim, transformer_hidden_dim) -> None:
        super().__init__()
        
        # Posterior network: outputs mean and log variance
        self.post_mean_head = nn.Linear(state_feat_dim, state_feat_dim)
        self.post_logvar_head = nn.Linear(state_feat_dim, state_feat_dim)
        
        # Prior network: outputs mean and log variance  
        self.prior_mean_head = nn.Linear(transformer_hidden_dim, state_feat_dim)
        self.prior_logvar_head = nn.Linear(transformer_hidden_dim, state_feat_dim)
        
        # Initialize small variance for stability
        self.min_logvar = -10
        self.max_logvar = 2

    def forward_post(self, x):
        mean = self.post_mean_head(x)
        logvar = self.post_logvar_head(x)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def forward_prior(self, x):
        mean = self.prior_mean_head(x)
        logvar = self.prior_logvar_head(x)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    

class GaussianKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_mean, p_logvar, q_mean, q_logvar):
        """
        Compute KL divergence between two Gaussian distributions:
        KL(p || q) where p is posterior, q is prior
        """
        # KL divergence formula for Gaussian distributions
        kl_div = 0.5 * (q_logvar - p_logvar + 
                        (torch.exp(p_logvar) + (p_mean - q_mean)**2) / torch.exp(q_logvar) - 1)
        
        # Sum over feature dimension, average over batch and sequence
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        
        real_kl_div = kl_div
        # Apply free bits constraint
        kl_div = torch.max(torch.ones_like(kl_div) * self.free_bits, kl_div)
        return kl_div, real_kl_div
    

class TerminationDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        return termination
    

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L D -> B L", "sum")
        return loss.mean()


class WorldModel(nn.Module):
    def __init__(self, args, state_dim, action_dim, state_feat_dim, task_embed_size, 
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, 
                 penalty_coeff=0, ensemble_size=1, stochasity=False):
        super().__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_feat_dim = state_feat_dim
        self.transformer_hidden_dim = transformer_hidden_dim

        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.num_layers = 2
        self.hidden_size = transformer_hidden_dim
        self.stochasity = stochasity
        self.penalty_coeff = penalty_coeff

        self.state_reward_encoder = FlattenMlp(input_size=state_dim+1,
                            output_size=state_feat_dim,
                            hidden_sizes=[self.hidden_size for _ in range(self.num_layers)])


        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=state_feat_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0,
            task_embedding_size=task_embed_size
        )

        self.ensemble_size = ensemble_size
        self.state_reward_decoder = models.GaussianMLP(in_size=state_feat_dim,
                                                  out_size=state_dim+1,
                                                  device=ptu.device,
                                                  num_layers=self.num_layers,
                                                  ensemble_size=self.ensemble_size,
                                                  hid_size=self.hidden_size,
                                                  learn_logvar_bounds=True,
                                                  deterministic=False).requires_grad_(True)
        
        self.termination_decoder = TerminationDecoder(
            transformer_hidden_dim=transformer_hidden_dim
        )
        
        self.dist_head = DistHead(
            state_feat_dim=state_feat_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )

        self.gaussian_kl_div_loss = GaussianKLDivLossWithFreeBits(free_bits=1)
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()


        if self.stochasity:
            self.log_var = nn.Parameter(torch.zeros(1, task_embed_size), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.best_snapshot_loss = 1e10

    
    # metric loss in FOCAL
    def metric_loss(self, z, tasks, epsilon=1e-3):
        # z shape is (task, dim)
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(tasks)):
            for j in range(i+1, len(tasks)):
                # positive pair
                if tasks[i] == tasks[j]:
                    pos_z_loss += torch.sqrt(torch.mean((z[i] - z[j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((z[i] - z[j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

    
    def predic_next_with_context(self, state, action, pre_reward, deterministic=False):
        T, L, _ = state.shape
        embedding = self.state_reward_encoder(state, pre_reward)
        post_mean, post_logvar = self.dist_head.forward_post(embedding)
        post_sample = self.dist_head.reparameterize(post_mean, post_logvar)
        mask = get_subsequent_mask_with_batch_length(batch_length=L, device=ptu.device)
        feats = self.storm_transformer.predict_next_with_context(post_sample, action, mask) 
        prior_mean, prior_logvar = self.dist_head.forward_prior(feats[:, -1:, :]) # [T, 1, D]
        prior_sample = self.dist_head.reparameterize(prior_mean, prior_logvar)
        next_state_reward_hat_mean, next_state_reward_hat_logvar = self.state_reward_decoder(prior_sample.squeeze(1)) # [ensemble_size, T, D]

        if self.ensemble_size!=1:
            index = torch.argmin(next_state_reward_hat_logvar.mean((1, 2)))
            next_state_reward_hat_mean = next_state_reward_hat_mean[index]
            next_state_reward_hat_logvar = next_state_reward_hat_logvar[index]

        next_state_reward_hat_std = torch.exp(0.5 * next_state_reward_hat_logvar)
        if deterministic:
            next_state_reward_hat = next_state_reward_hat_mean
        else:
            # sample next state
            next_state_reward_hat_eps = torch.randn_like(next_state_reward_hat_std)
            next_state_reward_hat = next_state_reward_hat_mean + next_state_reward_hat_eps * next_state_reward_hat_std

        next_state_hat = next_state_reward_hat[..., :-1]
        reward_hat = next_state_reward_hat[..., -1:]

        if self.penalty_coeff!=0:
            # penalty rewards
            penalty = torch.linalg.norm(next_state_reward_hat_std, dim=-1, keepdim=True)
            penalized_reward = reward_hat - self.penalty_coeff * penalty
            reward_hat = penalized_reward

        next_state_hat = next_state_hat.unsqueeze(1)
        reward_hat = reward_hat.unsqueeze(1)
        termination_hat = self.termination_decoder(feats[:, -1:, :])
        termination_hat = termination_hat > 0

        return next_state_hat, reward_hat, termination_hat

    def predict_next_with_kv_cache(self, post_sample, action, deterministic=False):
        # post_sample, action, pre_reward: [B, 1, D]
        feats = self.storm_transformer.forward_with_kv_cache(post_sample, action)

        # decoding
        prior_mean, prior_logvar = self.dist_head.forward_prior(feats.squeeze(1))
        prior_sample = self.dist_head.reparameterize(prior_mean, prior_logvar)
        next_state_reward_hat_mean, next_state_reward_hat_logvar = self.state_reward_decoder(prior_sample)
        
        if self.ensemble_size!=1:
            index = torch.argmin(next_state_reward_hat_logvar.mean((1, 2)))
            next_state_reward_hat_mean = next_state_reward_hat_mean[index]
            next_state_reward_hat_logvar = next_state_reward_hat_logvar[index]

        next_state_reward_hat_std = torch.exp(0.5 * next_state_reward_hat_logvar)
            
        
        if deterministic:
            next_state_reward_hat = next_state_reward_hat_mean
        else:
            next_state_reward_hat_eps = torch.randn_like(next_state_reward_hat_std)
            next_state_reward_hat = next_state_reward_hat_mean + next_state_reward_hat_eps * next_state_reward_hat_std
        
        next_state_hat = next_state_reward_hat[:, :-1]
        reward_hat = next_state_reward_hat[:, -1:]
        if self.penalty_coeff!=0:
            # penalty rewards
            penalty = torch.linalg.norm(next_state_reward_hat_std, dim=-1, keepdim=True)
            penalized_reward = reward_hat - self.penalty_coeff * penalty
            reward_hat = penalized_reward

        next_state_hat = next_state_hat.unsqueeze(1)
        reward_hat = reward_hat.unsqueeze(1)
        prior_sample = prior_sample.unsqueeze(1)
        termination_hat = self.termination_decoder(feats)
        termination_hat = termination_hat > 0

        return next_state_hat, reward_hat, termination_hat, prior_sample
    

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            state_size = (imagine_batch_size, imagine_batch_length+1, self.state_dim)
            reward_size = (imagine_batch_size, imagine_batch_length+1, 1)
            termination_size = (imagine_batch_size, imagine_batch_length, 1)
            action_size = (imagine_batch_size, imagine_batch_length, self.action_dim)
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.state_feat_dim)
            self.state_buffer = torch.zeros(state_size, device=ptu.device)
            self.action_buffer = torch.zeros(action_size, device=ptu.device)
            self.reward_hat_buffer = torch.zeros(reward_size, device=ptu.device)
            self.termination_hat_buffer = torch.zeros(termination_size, device=ptu.device)
            self.latent_buffer = torch.zeros(latent_size, device=ptu.device)


    def imagine_data(self, agent, sample_state, sample_action, sample_pre_reward, task_embedding,
                     imagine_batch_size, imagine_batch_length):
        # sample_state: (B, context_length=8, state_dim)
        # task_embedding: (B, 1, task_embed_size)
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length)
        self.storm_transformer.reset_kv_cache_list(imagine_batch_size)

        embedding = self.state_reward_encoder(sample_state, sample_pre_reward)
        post_mean, post_logvar = self.dist_head.forward_post(embedding)
        post_sample = self.dist_head.reparameterize(post_mean, post_logvar)

        # context
        for i in range(sample_state.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent_sample = self.predict_next_with_kv_cache(
                post_sample[:, i:i+1],
                sample_action[:, i:i+1]
            )
            
        self.state_buffer[:, 0:1] = last_obs_hat
        self.reward_hat_buffer[:, 0:1] = last_reward_hat
        self.latent_buffer[:, 0:1] = last_latent_sample


        # imagine
        for i in range(imagine_batch_length):
            augmented_obs = torch.cat([self.state_buffer[:, i:i+1], task_embedding], dim=-1)
            if self.args.policy == 'dqn':
                action, value = agent.act(obs=augmented_obs)
            else:
                action, _, _, _ = agent.act(obs=augmented_obs)
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent_sample = self.predict_next_with_kv_cache(
                self.latent_buffer[:, i:i+1], 
                self.action_buffer[:, i:i+1]
            )

            self.state_buffer[:, i+1:i+2] = last_obs_hat
            self.reward_hat_buffer[:, i+1:i+2] = last_reward_hat
            self.latent_buffer[:, i+1:i+2] = last_latent_sample
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat

        state_hat = self.state_buffer[:, :-1, :]
        next_state_hat = self.state_buffer[:, 1:, :]
        action_hat = self.action_buffer
        reward_hat = self.reward_hat_buffer[:, 1:, :]
        termination_hat = self.termination_hat_buffer
        pre_reward_hat = self.reward_hat_buffer[:, :-1, :]

        return state_hat, action_hat, reward_hat, next_state_hat, termination_hat, pre_reward_hat
    

    def reconstruction_loss(self, sample, target_state, target_reward):
        T, L, _ = sample.shape
        sample = sample.reshape(T * L, -1)
        target_state = target_state.reshape(T * L, -1)
        target_reward = target_reward.reshape(T * L, -1)
        target = torch.cat([target_state, target_reward], dim=-1)
        # predict next state
        sample = sample.repeat(self.ensemble_size, 1, 1)
        reconstruction_loss, _ = self.state_reward_decoder.loss(sample, target.repeat(self.ensemble_size, 1, 1))
        reconstruction_loss = reconstruction_loss / self.ensemble_size

        return reconstruction_loss

    
    def context_encoding(self, states, actions, pre_rewards):
        T, L, _ = states.shape
        embedding = self.state_reward_encoder(states, pre_rewards)
        post_mean, post_logvar = self.dist_head.forward_post(embedding)
        sample = self.dist_head.reparameterize(post_mean, post_logvar)
        mask = get_subsequent_mask_with_batch_length(batch_length=L+1, device=ptu.device)
        feats, task_encoding = self.storm_transformer(sample, actions, mask)
        
        if self.stochasity:
            var = self.log_var.expand_as(task_encoding).exp()
            clamped_var = torch.clamp(var, 0.1, 10.0)
            prob_dist = dist.Normal(task_encoding, clamped_var.sqrt())
            task_encoding = prob_dist.rsample()

        return task_encoding
    
    def update(self, tasks, states, actions, pre_rewards, terms):
        T, L, _ = states.shape
        embedding = self.state_reward_encoder(states, pre_rewards)
        post_mean, post_logvar = self.dist_head.forward_post(embedding)
        post_sample = self.dist_head.reparameterize(post_mean, post_logvar)

        mask = get_subsequent_mask_with_batch_length(batch_length=L+1, device=ptu.device)
        feats, task_encoding = self.storm_transformer(post_sample, actions, mask)
        
        if self.stochasity:
            var = self.log_var.expand_as(task_encoding).exp()
            clamped_var = torch.clamp(var, 0.1, 10.0)
            prob_dist = dist.Normal(task_encoding, clamped_var.sqrt())
            task_encoding = prob_dist.rsample()

        prior_mean, prior_logvar = self.dist_head.forward_prior(feats)

        # reconstruction_loss
        reconstruction_loss = self.reconstruction_loss(post_sample, states, pre_rewards)

        termination_hat = self.termination_decoder(feats)
        termination_loss = self.bce_with_logits_loss_func(termination_hat, terms)
        # dyn-rep loss - use Gaussian KL divergence
        # Note: we use [:, 1:] for posterior and [:, :-1] for prior to align sequences
        dynamics_loss, dynamics_real_kl_div = self.gaussian_kl_div_loss(
            post_mean[:, 1:].detach(), post_logvar[:, 1:].detach(),
            prior_mean[:, :-1], prior_logvar[:, :-1]
        )
        representation_loss, representation_real_kl_div = self.gaussian_kl_div_loss(
            post_mean[:, 1:], post_logvar[:, 1:],
            prior_mean[:, :-1].detach(), prior_logvar[:, :-1].detach()
        )

        # metric_loss = self.metric_loss(task_encoding, tasks)
        metric_loss = self.metric_loss(task_encoding, tasks)

        total_loss = reconstruction_loss + termination_loss + dynamics_loss + 0.1*representation_loss + metric_loss
        # gradient descent
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        wm_loss = {}
        wm_loss["WorldModel/reconstruction_loss"] = reconstruction_loss.item()
        wm_loss["WorldModel/dynamics_loss"] = dynamics_loss.item()
        wm_loss["WorldModel/termination_loss"] = termination_loss.item()
        wm_loss["WorldModel/metric_loss"] = metric_loss.item()
        wm_loss["WorldModel/total_loss"] = total_loss.item()

        return task_encoding, wm_loss
    

    def eval(self, tasks, states, actions, pre_rewards, terms):
        with torch.no_grad():
            T, L, _ = states.shape
            embedding = self.state_reward_encoder(states, pre_rewards)
            post_mean, post_logvar = self.dist_head.forward_post(embedding)
            post_sample = self.dist_head.reparameterize(post_mean, post_logvar)

            mask = get_subsequent_mask_with_batch_length(batch_length=L+1, device=ptu.device)
            feats, task_encoding = self.storm_transformer(post_sample, actions, mask)
            
            if self.stochasity:
                var = self.log_var.expand_as(task_encoding).exp()
                clamped_var = torch.clamp(var, 0.1, 10.0)
                prob_dist = dist.Normal(task_encoding, clamped_var.sqrt())
                task_encoding = prob_dist.rsample()

            prior_mean, prior_logvar = self.dist_head.forward_prior(feats)

            # reconstruction_loss
            reconstruction_loss = self.reconstruction_loss(post_sample, states, pre_rewards)
            

            termination_hat = self.termination_decoder(feats)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, terms)
            # dyn-rep loss - use Gaussian KL divergence
            # Note: we use [:, 1:] for posterior and [:, :-1] for prior to align sequences
            dynamics_loss, dynamics_real_kl_div = self.gaussian_kl_div_loss(
                post_mean[:, 1:].detach(), post_logvar[:, 1:].detach(),
                prior_mean[:, :-1], prior_logvar[:, :-1]
            )
            representation_loss, representation_real_kl_div = self.gaussian_kl_div_loss(
                post_mean[:, 1:], post_logvar[:, 1:],
                prior_mean[:, :-1].detach(), prior_logvar[:, :-1].detach()
            )

            metric_loss = self.metric_loss(task_encoding, tasks)
 
            total_loss = reconstruction_loss + termination_loss + dynamics_loss + 0.1*representation_loss + metric_loss
            
            wm_loss = {}
            wm_loss["PretrainWorldModel/reconstruction_loss"] = reconstruction_loss.item()
            wm_loss["PretrainWorldModel/dynamics_loss"] = dynamics_loss.item()
            wm_loss["PretrainWorldModel/termination_loss"] = termination_loss.item()
            wm_loss["PretrainWorldModel/metric_loss"] = metric_loss.item()
            wm_loss["PretrainWorldModel/total_loss"] = total_loss.item()

        return wm_loss
    

    def update_best_snapshot(self, val_loss):
        updated = False
        current_loss = val_loss
        best_loss = self.best_snapshot_loss
        if best_loss - current_loss > 0:
            self.best_snapshot_loss = current_loss
            updated = True

        return updated
    

    def load_best_snapshot(self, path):
        snapshot = torch.load(path, map_location=ptu.device)
        self.load_state_dict(snapshot)
    
