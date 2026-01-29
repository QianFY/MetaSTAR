# offline meta rl with contrastive representation learning 
# encoder and agent training are disentangled
# FOCAL: sample pos/neg pairs from same/diff task replay buffers
# relabel-gt: sample neg pairs with gt reward/state relabelling
# relabel-separate: learn reward/transition models for each task, sample neg pairs with the learned relabelling models
# ours: learn conditional generative model over all tasks, sample neg pairs with the learned generative model

import os
import sys
import time
import argparse
import torch
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_cheetah_vel, args_cheetah_dir, args_ant_dir, args_hopper_param,\
	  args_walker_param, args_cheetah_vel_sparse, args_point_robot_sparse, args_ant_semicircle, \
		args_walker_param_sparse, args_hopper_param_sparse
import numpy as np

from algorithms.dqn import DQN
from algorithms.combo import COMBO
from environments.make_env import make_env
import torchkit.pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy

from utils.visual_offline_dataset import visual_by_tsne
import copy
# import mbrl

from collections import deque


from wm_models.world_models import WorldModel
import torch.nn.functional as F




class MetaSTAR:
	# algorithm class of offline meta-rl with contrastive learning
	# training: (learn models), sample pos/neg pairs (with relabelling), train encoder, train dqn/sac
	# testing: given task context set, extract task encoding, rollout policy in the env

	def __init__(self, args, train_dataset, train_goals, eval_dataset, eval_goals):
		"""
		Seeds everything.
		Initialises: logger, environments, policy (+storage +optimiser).
		"""

		self.args = args
		self.context_size = self.args.trajectory_len // 2
		self.ImagineContextLength = self.args.imagine_context_length
		self.ImagineBatchLength = self.args.imagine_length
		self.rollouts_per_step = self.args.rollouts_per_step
		self.pretrain_wm_iters = 10000

		# make sure everything has the same seed
		utl.seed(self.args.seed)

		# initialize tensorboard logger 
		if self.args.log_tensorboard:
			self.tb_logger = TBLogger(self.args)

		self.args, _ = off_utl.expand_args(self.args, include_act_space=True)
		if self.args.act_space.__class__.__name__ == "Discrete":
			self.args.policy = 'dqn'
		else:
			self.args.policy = 'sac'
		
		
		if self.args.pearl_deterministic_encoder:  
			self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size
		else:
			self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size * 2

		self.goals = train_goals
		self.eval_goals = eval_goals
		# context set, to extract task encoding
		self.context_dataset = train_dataset       
		self.eval_context_dataset = eval_dataset

		self.storage = self.load_buffer(train_dataset, train_goals)
		self.eval_storage = self.load_buffer(eval_dataset, eval_goals)

		# initialize policy
		self.initialize_policy()
		# initialize task encoder
		self.initialize_world_model()


		# create environment for evaluation    
		self.env = make_env(args.env_name,
							args.max_rollouts_per_task,
							seed=args.seed,
							n_tasks=self.args.num_eval_tasks)
		# fix the possible eval goals to be the testing set's goals
		self.env.set_all_goals(eval_goals)
		self.env_train = make_env(args.env_name,
							args.max_rollouts_per_task,
							seed=args.seed,
							n_tasks=self.args.num_train_tasks) 
		self.env_train.set_all_goals(train_goals)


	def initialize_world_model(self):
		self.world_model = WorldModel(
			args = self.args,
			state_dim=self.args.obs_dim,
			action_dim=self.args.action_dim,
			state_feat_dim=self.args.state_feat_dim,
			task_embed_size=self.args.task_embedding_size,
			transformer_max_length=self.context_size,
			transformer_hidden_dim=self.args.transformer_hidden_dim,
			transformer_num_layers=self.args.transformer_num_layers,
			transformer_num_heads=self.args.transformer_num_heads
		).to(ptu.device)
		# world_model_params = sum(p.numel() for p in self.world_model.parameters())
		# print(f'World Model Parameters: {world_model_params}')


	def initialize_policy(self):
		if self.args.policy == 'dqn':
			q_network = FlattenMlp(input_size=self.args.augmented_obs_dim,
								   output_size=self.args.act_space.n,
								   hidden_sizes=self.args.dqn_layers)
			self.agent = DQN(
				q_network,
				# optimiser_vae=self.optimizer_vae,
				lr=self.args.policy_lr,
				gamma=self.args.gamma,
				tau=self.args.soft_target_tau,
			).to(ptu.device)
		else:
			# assert self.args.act_space.__class__.__name__ == "Box", (
			#     "Can't train SAC with discrete action space!")
			q1_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
									output_size=1,
									hidden_sizes=self.args.dqn_layers)
			q2_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
									output_size=1,
									hidden_sizes=self.args.dqn_layers)

			policy = TanhGaussianPolicy(obs_dim=self.args.augmented_obs_dim,
										action_dim=self.args.action_dim,
										hidden_sizes=self.args.policy_layers)
		
			self.agent = COMBO(
				policy,
				q1_network,
				q2_network,
				actor_lr=self.args.actor_lr,
				critic_lr=self.args.critic_lr,
				gamma=self.args.gamma,
				tau=self.args.soft_target_tau,
				use_cql=self.args.use_cql if 'use_cql' in self.args else False,
				alpha_cql=self.args.alpha_cql if 'alpha_cql' in self.args else None,
				entropy_alpha=self.args.entropy_alpha,
				automatic_entropy_tuning=self.args.automatic_entropy_tuning,
				alpha_lr=self.args.alpha_lr,
				clip_grad_value=self.args.clip_grad_value,
			).to(ptu.device)
	

	# convert the training set to the multitask replay buffer
	def load_buffer(self, dataset, goals):
		# process obs, actions, ... into shape (num_trajs*num_timesteps, dim) for each task
		dataset_list = []
		# total_transition_per_task = len(train_dataset[0][0]) * len(train_dataset[0][0][0])
		# visual = np.zeros((len(train_goals), total_transition_per_task, self.args.obs_dim * 2 + self.args.action_dim + 1 + 1))
		for i, set in enumerate(dataset):
			obs, actions, rewards, next_obs, terminals = set
			
			device=ptu.device
			obs = ptu.FloatTensor(obs).to(device)
			actions = ptu.FloatTensor(actions).to(device)
			rewards = ptu.FloatTensor(rewards).to(device)
			next_obs = ptu.FloatTensor(next_obs).to(device)
			terminals = ptu.FloatTensor(terminals).to(device)
			pre_rewards = rewards[:-1, :, :] # [time_step, num_traj, 1]
			pre_rewards = torch.cat([torch.zeros_like(pre_rewards[:1, :, :]), pre_rewards], dim=0)

			obs = obs.transpose(0, 1).reshape(-1, obs.shape[-1])
			actions = actions.transpose(0, 1).reshape(-1, actions.shape[-1])
			rewards = rewards.transpose(0, 1).reshape(-1, rewards.shape[-1])
			next_obs = next_obs.transpose(0, 1).reshape(-1, next_obs.shape[-1])
			terminals = terminals.transpose(0, 1).reshape(-1, terminals.shape[-1])
			pre_rewards = pre_rewards.transpose(0, 1).reshape(-1, pre_rewards.shape[-1])

			obs = ptu.get_numpy(obs)
			actions = ptu.get_numpy(actions)
			rewards = ptu.get_numpy(rewards)
			next_obs = ptu.get_numpy(next_obs)
			terminals = ptu.get_numpy(terminals)
			pre_rewards = ptu.get_numpy(pre_rewards)

			dataset_list.append([obs, actions, rewards, next_obs, terminals, pre_rewards])
			


		storage = MultiTaskPolicyStorage(max_replay_buffer_size=dataset_list[0][0].shape[0],
											  obs_dim=dataset_list[0][0].shape[1],
											  action_space=self.args.act_space,
											  tasks=range(len(goals)),
											  trajectory_len=self.args.trajectory_len)

		for task, set in enumerate(dataset_list):  
			storage.add_samples(task,
									 observations=set[0],
									 actions=set[1],
									 rewards=set[2],
									 next_observations=set[3],
									 terminals=set[4],
									 pre_rewards=set[5])  
		return storage
	
	def pretrain_world_model(self):
		# pre-train world model only
		print('pre-training world model...')
		num_epochs_since_prev_best = 0
		
		for pre_iter in range(1, self.pretrain_wm_iters+1):
			obs_context, actions_context, rewards_context, next_obs_context, terms_context, pre_rewards_context = self.sample_random_context_batch(range(len(self.goals)), self.context_size)

			_, wm_loss = self.world_model.update(range(len(self.goals)), obs_context, actions_context, pre_rewards_context, terms_context)

			obs_context_eval, actions_context_eval, rewards_context_eval, next_obs_context_eval, terms_context_eval, pre_rewards_context_eval = self.sample_random_context_batch(range(len(self.eval_goals)), self.context_size, trainset=False)
			wm_loss_eval = self.world_model.eval(range(len(self.eval_goals)), obs_context_eval, actions_context_eval, pre_rewards_context_eval, terms_context_eval)
			update = self.world_model.update_best_snapshot(wm_loss_eval['PretrainWorldModel/total_loss'])
	
			num_epochs_since_prev_best += 1
			if update:
				num_epochs_since_prev_best = 0
				save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
				if not os.path.exists(save_path):
					os.mkdir(save_path)
				torch.save(self.world_model.state_dict(), os.path.join(save_path, "world_model_pretrain.pt"))
			print(f'Pre-train WM Iteration {pre_iter}, WM Train Loss: {wm_loss["WorldModel/total_loss"]}, WM Eval Loss: {wm_loss_eval["PretrainWorldModel/total_loss"]}')
			for k in wm_loss_eval.keys():
				self.tb_logger.writer.add_scalar(k, wm_loss_eval[k], pre_iter)
			# if num_epochs_since_prev_best >= self.args.rl_updates_per_iter:
			if num_epochs_since_prev_best >= 1000:
				print('early stopping wm pre-training at iter {}'.format(pre_iter))
				break
		print('pre-training world model done.')
		self.world_model.load_best_snapshot(os.path.join(save_path, "world_model_pretrain.pt"))
		
	
	def world_model_imagine_data(self, tasks, task_encoding):
		with torch.no_grad():
			total_batch_size = len(tasks) * self.rollouts_per_step
			expanded_tasks = [t for t in tasks for _ in range(self.rollouts_per_step)]
			expanded_task_encoding = task_encoding.repeat_interleave(self.rollouts_per_step, dim=0)
		
			sample_obs, sample_actions, sample_rewards, sample_next_obs, sample_terms, sample_pre_rewards = self.sample_random_context_batch(expanded_tasks, self.ImagineContextLength)

			imagine_obs, imagine_actions, imagine_rewards, imagine_next_obs, imagine_terms, imagine_pre_rewards = self.world_model.imagine_data(self.agent,
				sample_obs, sample_actions, sample_pre_rewards, expanded_task_encoding,
				total_batch_size, self.ImagineBatchLength)
				
			def reshape_to_target(x):
				x = x.view(len(tasks), self.rollouts_per_step, self.ImagineBatchLength, -1)
				return x.flatten(1, 2)  # [task, self.rollouts_per_step*self.ImagineBatchLength, dim]
			
			obs = reshape_to_target(imagine_obs)
			actions = reshape_to_target(imagine_actions)
			rewards = reshape_to_target(imagine_rewards)
			next_obs = reshape_to_target(imagine_next_obs)
			terms = reshape_to_target(imagine_terms)
			pre_rewards = reshape_to_target(imagine_pre_rewards)
			
			return obs, actions, rewards, next_obs, terms, pre_rewards
		


	# training offline RL, with evaluation on fixed eval tasks
	def train(self):
		self._start_training()
		#print('start training')
		self.pretrain_world_model()

		for iter_ in range(self.args.num_iters):   
			self.training_mode(True)
			indices = np.random.choice(len(self.goals), self.args.meta_batch) # sample with replacement! it is important for FOCAL
		
			#print('training')
			train_stats = self.update(iter_, indices)

			self.training_mode(False)
			#print('logging')
			self.log(iter_ + 1, train_stats)


	def update(self, iteration, tasks):
		rl_losses_agg = {}
		time_cost = {'data_sampling':0, 'update_encoder':0, 'update_rl':0}

		for update in range(self.args.rl_updates_per_iter): 
			if self.args.log_train_time:
				_t_cost = time.time()

			obs_context, actions_context, rewards_context, next_obs_context, terms_context, pre_rewards_context = self.sample_random_context_batch(tasks, self.context_size)

			# task_encoding, wm_loss = self.world_model.update(tasks, obs_context, actions_context, pre_rewards_context, terms_context)
			task_encoding, wm_loss = self.world_model.update(tasks, obs_context, actions_context, pre_rewards_context, terms_context)

			task_encoding = task_encoding.detach().unsqueeze(1)
			t, _, d = task_encoding.size()

			model_batch_size = self.rollouts_per_step * self.ImagineBatchLength
			offline_batch_size = self.args.rl_batch_size - model_batch_size
			obs, actions, rewards, next_obs, terms, pre_rewards = self.sample_rl_batch(tasks, offline_batch_size)
			if self.rollouts_per_step != 0:
				model_obs, model_actions, model_rewards, model_next_obs, model_terms, _ = self.world_model_imagine_data(tasks, task_encoding)
				
				obs = torch.cat((obs, model_obs), dim=1)
				actions = torch.cat((actions, model_actions), dim=1)
				rewards = torch.cat((rewards, model_rewards), dim=1)
				next_obs = torch.cat((next_obs, model_next_obs), dim=1)
				terms = torch.cat((terms, model_terms), dim=1)
			
			task_encoding = task_encoding.expand(t, self.args.rl_batch_size, d) # [task, batch(repeat), dim]

			obs = torch.cat((obs, task_encoding), dim=-1)
			next_obs = torch.cat((next_obs, task_encoding), dim=-1) # [task, batch, obs_dim+z_dim]

			real_obs = obs[:, :offline_batch_size, :]
			real_actions = actions[:, :offline_batch_size, :]
			
			t, b, _ = obs.shape
			obs = obs.view(t * b, -1)
			actions = actions.view(t * b, -1)
			rewards = rewards.view(t * b, -1)
			next_obs = next_obs.view(t * b, -1)
			terms = terms.view(t * b, -1)
			

			real_obs = real_obs.reshape(t * offline_batch_size, -1)
			real_actions = real_actions.reshape(t * offline_batch_size, -1)

		
			rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms, real_obs, real_actions)
				
			
			if self.args.log_train_time:
				_t_now = time.time()
				time_cost['update_rl'] += (_t_now-_t_cost)
				_t_cost = _t_now

			rl_losses = {f"rl_losses/{k}": v for k, v in rl_losses.items()}

			total_loss = wm_loss | rl_losses 
			

			for k, v in total_loss.items():
				if update == 0:  # first iterate - create list
					rl_losses_agg[k] = [v]
				else:  # append values
					rl_losses_agg[k].append(v)

		# take mean
		for k in rl_losses_agg:
			rl_losses_agg[k] = np.mean(rl_losses_agg[k])
		self._n_rl_update_steps_total += self.args.rl_updates_per_iter

		if self.args.log_train_time:
			print(time_cost)

		return rl_losses_agg
			

	# do policy evaluation on eval tasks
	def evaluate(self, trainset=False, offline=True):
		if offline:
			num_episodes = self.args.max_rollouts_per_task
		else:
			num_episodes = self.args.online_rollouts_per_task      
		num_steps_per_episode = self.env.unwrapped._max_episode_steps
		num_tasks = self.args.num_train_tasks if trainset else self.args.num_eval_tasks 
		obs_size = self.env.unwrapped.observation_space.shape[0]

		returns_per_episode = np.zeros((num_tasks, num_episodes))
		success_rate = np.zeros(num_tasks)

		rewards = np.zeros((num_tasks, self.args.trajectory_len))
		reward_preds = np.zeros((num_tasks, self.args.trajectory_len))
		observations = np.zeros((num_tasks, self.args.trajectory_len + 1, obs_size))
		if self.args.policy == 'sac':
			log_probs = np.zeros((num_tasks, self.args.trajectory_len))

		eval_env = self.env_train if trainset else self.env
		
		for task in eval_env.unwrapped.get_all_task_idx():

			
			if offline:
				obs_context, actions_context, rewards_context, next_obs_context, terms_context, pre_rewards_context = self.sample_random_context_batch([task], self.context_size, trainset=trainset)
				task_desc = self.world_model.context_encoding(states=obs_context, actions=actions_context, pre_rewards=pre_rewards_context)  

			else:
				obs_context_queue = deque(maxlen=self.context_size)
				actions_context_queue = deque(maxlen=self.context_size)
				pre_rewards_context_queue = deque(maxlen=self.context_size)


			for episode_idx in range(num_episodes):
				obs = ptu.from_numpy(eval_env.reset(task))
				obs = obs.reshape(-1, obs.shape[-1])
				step = 0      
				observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

				running_reward = 0.
				reward = torch.zeros(1, 1).to(ptu.device)
				for step_idx in range(num_steps_per_episode):
					# add distribution parameters to observation - policy is conditioned on posterior
					if offline:
						augmented_obs = torch.cat((obs, task_desc), dim=-1)
						if self.args.policy == 'dqn':
							action, value = self.agent.act(obs=augmented_obs, deterministic=True)
						else:
							action, _, _, log_prob = self.agent.act(obs=augmented_obs,
																	deterministic=self.args.eval_deterministic,
																	return_log_prob=True)
							log_probs[task, step] = ptu.get_numpy(log_prob[0])
					else:
						if len(obs_context_queue) < self.context_size:
							if self.args.policy == 'dqn':
								action = torch.randint(0, self.args.act_space.n, (1, 1)).to(ptu.device)
							else:
								action = torch.FloatTensor(1, self.args.action_dim).uniform_(-1.0, 1.0).to(ptu.device)
						else:
							obs_context = torch.cat(list(obs_context_queue), dim=0).unsqueeze(0)
							actions_context = torch.cat(list(actions_context_queue), dim=0).unsqueeze(0)
							pre_rewards_context = torch.cat(list(pre_rewards_context_queue), dim=0).unsqueeze(0)
							task_desc = self.world_model.context_encoding(states=obs_context,
																			actions=actions_context,
																			pre_rewards=pre_rewards_context)
							augmented_obs = torch.cat((obs, task_desc), dim=-1)
							if self.args.policy == 'dqn':
								action, value = self.agent.act(obs=augmented_obs, deterministic=True)
							else:
								action, _, _, log_prob = self.agent.act(obs=augmented_obs,
																		deterministic=self.args.eval_deterministic,
																		return_log_prob=True)
								log_probs[task, step] = ptu.get_numpy(log_prob[0])

				
						obs_context_queue.append(obs[:, :obs_size])
						actions_context_queue.append(action)
						pre_rewards_context_queue.append(reward)

					# observe reward and next obs  
					next_obs, reward, done, info = utl.env_step(eval_env, action.squeeze(dim=0))


					running_reward += reward.item()

					rewards[task, step] = reward.item()

					observations[task, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])

					if "is_goal_state" in dir(eval_env.unwrapped) and eval_env.unwrapped.is_goal_state():
						success_rate[task] = 1.
					# set: obs <- next_obs
					obs = next_obs.clone()
					step += 1

				returns_per_episode[task, episode_idx] = running_reward

		# reward_preds is 0 here
		if self.args.policy == 'dqn':
			return returns_per_episode[:, -1], success_rate, observations, rewards, reward_preds
		else:
			return returns_per_episode[:, -1], success_rate, log_probs, observations, rewards, reward_preds
		

	def evaluate4view(self, num_episodes=1, n_tasks=15, offline=True):
  
		num_steps_per_episode = self.env.unwrapped._max_episode_steps
		num_tasks = self.args.num_eval_tasks 
		if num_tasks > n_tasks:
			num_tasks = n_tasks
		obs_size = self.env.unwrapped.observation_space.shape[0]

		returns_per_episode = np.zeros((num_tasks, num_episodes))
		success_rate = np.zeros(num_tasks)

		rewards = np.zeros((num_tasks, self.args.trajectory_len))
		reward_preds = np.zeros((num_tasks, self.args.trajectory_len))
		observations = np.zeros((num_tasks, self.args.trajectory_len + 1, obs_size))
		if self.args.policy == 'sac':
			log_probs = np.zeros((num_tasks, self.args.trajectory_len))

		eval_env = self.env
		task_desc_per_step = np.zeros((num_tasks, self.args.trajectory_len, self.args.task_embedding_size))
		# task_desc_per_step = np.zeros((num_tasks, self.context_size+(num_episodes-1)*self.args.trajectory_len, self.args.task_embedding_size))
		
		tasks = eval_env.unwrapped.get_all_task_idx()
		if len(tasks) > n_tasks:
			tasks = tasks[:n_tasks]
		for task in tasks:

			
			if offline:
				obs_context, actions_context, rewards_context, next_obs_context, terms_context, pre_rewards_context = self.sample_random_context_batch([task], self.context_size, trainset=False)
				task_desc = self.world_model.context_encoding(states=obs_context, actions=actions_context, pre_rewards=pre_rewards_context)  

			else:
				obs_context_queue = deque(maxlen=self.context_size)
				actions_context_queue = deque(maxlen=self.context_size)
				pre_rewards_context_queue = deque(maxlen=self.context_size)


			for episode_idx in range(num_episodes):
				obs = ptu.from_numpy(eval_env.reset(task))
				obs = obs.reshape(-1, obs.shape[-1])
				step = 0      
				observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

				running_reward = 0.
				reward = torch.zeros(1, 1).to(ptu.device)
				for step_idx in range(num_steps_per_episode):
					# add distribution parameters to observation - policy is conditioned on posterior
					if offline:
						augmented_obs = torch.cat((obs, task_desc), dim=-1)
						if self.args.policy == 'dqn':
							action, value = self.agent.act(obs=augmented_obs, deterministic=True)
						else:
							action, _, _, log_prob = self.agent.act(obs=augmented_obs,
																	deterministic=self.args.eval_deterministic,
																	return_log_prob=True)
							log_probs[task, step] = ptu.get_numpy(log_prob[0])
					else:
						if len(obs_context_queue) < self.context_size:
							if self.args.policy == 'dqn':
								action = torch.randint(0, self.args.act_space.n, (1, 1)).to(ptu.device)
							else:
								action = torch.FloatTensor(1, self.args.action_dim).uniform_(-1.0, 1.0).to(ptu.device)
						else:
							obs_context = torch.cat(list(obs_context_queue), dim=0).unsqueeze(0)
							actions_context = torch.cat(list(actions_context_queue), dim=0).unsqueeze(0)
							pre_rewards_context = torch.cat(list(pre_rewards_context_queue), dim=0).unsqueeze(0)
							task_desc = self.world_model.context_encoding(states=obs_context,
																			actions=actions_context,
																			pre_rewards=pre_rewards_context)
							augmented_obs = torch.cat((obs, task_desc), dim=-1)
							if self.args.policy == 'dqn':
								action, value = self.agent.act(obs=augmented_obs, deterministic=True)
							else:
								action, _, _, log_prob = self.agent.act(obs=augmented_obs,
																		deterministic=self.args.eval_deterministic,
																		return_log_prob=True)
								log_probs[task, step] = ptu.get_numpy(log_prob[0])

							task_desc_per_step[task, step, :] = ptu.get_numpy(task_desc[0])
							# task_desc_per_step[task, episode_idx*num_steps_per_episode+step-self.context_size, :] = ptu.get_numpy(task_desc[0])			

						obs_context_queue.append(obs[:, :obs_size])
						actions_context_queue.append(action)
						pre_rewards_context_queue.append(reward)

					# observe reward and next obs  
					next_obs, reward, done, info = utl.env_step(eval_env, action.squeeze(dim=0))


					running_reward += reward.item()

					rewards[task, step] = reward.item()

					observations[task, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])

					if "is_goal_state" in dir(eval_env.unwrapped) and eval_env.unwrapped.is_goal_state():
						success_rate[task] = 1.
					# set: obs <- next_obs
					obs = next_obs.clone()
					step += 1

				returns_per_episode[task, episode_idx] = running_reward

		# reward_preds is 0 here
		if self.args.policy == 'dqn':
			return returns_per_episode[:, -1], success_rate, observations, rewards, reward_preds, returns_per_episode, task_desc_per_step
		else:
			return returns_per_episode[:, -1], success_rate, log_probs, observations, rewards, reward_preds, returns_per_episode, task_desc_per_step

	def load_parameter(self, agent_addr=None, world_model_addr=None):
		if agent_addr:
			self.agent.load_state_dict(torch.load(agent_addr, map_location=ptu.device))
		if world_model_addr:
			self.world_model.load_state_dict(torch.load(world_model_addr, map_location=ptu.device))


	def log(self, iteration, train_stats):
		# --- save model ---
		if iteration % self.args.save_interval == 0:
			save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			torch.save(self.agent.state_dict(), os.path.join(save_path, "agent{0}.pt".format(iteration)))
			torch.save(self.world_model.state_dict(), os.path.join(save_path, "world_model{0}.pt".format(iteration)))

		if iteration % self.args.log_interval == 0 or iteration == 1:
			if self.args.policy == 'dqn':
				returns, success_rate, observations, rewards, reward_preds = self.evaluate()
				returns_train, success_rate_train, observations_train, rewards_train, reward_preds_train = self.evaluate(trainset=True)
				online_returns, online_success_rate, online_observations, online_rewards, online_reward_preds = self.evaluate(offline=False)

			else:   
				returns, success_rate, log_probs, observations, rewards, reward_preds = self.evaluate()
				returns_train, success_rate_train, log_probs_train, observations_train, rewards_train, reward_preds_train = self.evaluate(trainset=True)
				online_returns, online_success_rate, online_log_probs, online_observations, online_rewards, online_reward_preds = self.evaluate(offline=False)

			if self.args.log_tensorboard:
				if self.args.env_name == 'GridBlock-v2':
					tasks_to_vis = np.random.choice(self.args.num_eval_tasks, 5)
					for i, task in enumerate(tasks_to_vis):
						self.env.reset(task)
						self.tb_logger.writer.add_figure('policy_vis/task_{}'.format(i),
													 utl_eval.plot_rollouts(observations[task, :], self.env),
													 self._n_rl_update_steps_total)
						self.tb_logger.writer.add_figure('reward_prediction_train/task_{}'.format(i),
													 utl_eval.plot_rew_pred_vs_rew(rewards[task, :],
																				   reward_preds[task, :]),
													 self._n_rl_update_steps_total)

				if self.args.max_rollouts_per_task > 1:
					raise NotImplementedError
				else:   
					self.tb_logger.writer.add_scalar('returns/returns_mean', np.mean(returns),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns/returns_std', np.std(returns),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns/success_rate', np.mean(success_rate),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns_train/returns_mean', np.mean(returns_train),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns_train/returns_std', np.std(returns_train),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns_train/success_rate', np.mean(success_rate_train),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('online_returns/returns_mean', np.mean(online_returns),
														self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('online_returns/returns_std', np.std(online_returns),
														self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('online_returns/success_rate', np.mean(online_success_rate),
														self._n_rl_update_steps_total)
				if self.args.policy == 'dqn':
					for k in train_stats.keys():
						self.tb_logger.writer.add_scalar(k, train_stats[k], 
							self._n_rl_update_steps_total)

					self.tb_logger.writer.add_scalar('weights/q_network',
													 list(self.agent.qf.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf.parameters())[0].grad is not None:
						param_list = list(self.agent.qf.parameters())
						self.tb_logger.writer.add_scalar('gradients/q_network',
														 sum([param_list[i].grad.mean() for i in
															  range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q_target',
													 list(self.agent.target_qf.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.target_qf.parameters())[0].grad is not None:
						param_list = list(self.agent.target_qf.parameters())
						self.tb_logger.writer.add_scalar('gradients/q_target',
														 sum([param_list[i].grad.mean() for i in
															  range(len(param_list))]),
														 self._n_rl_update_steps_total)
				else:
					for k in train_stats.keys():
						self.tb_logger.writer.add_scalar(k, train_stats[k], 
							self._n_rl_update_steps_total)

					# weights and gradients
					self.tb_logger.writer.add_scalar('weights/q1_network',
													 list(self.agent.qf1.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf1.parameters())[0].grad is not None:
						param_list = list(self.agent.qf1.parameters())
						self.tb_logger.writer.add_scalar('gradients/q1_network',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q1_target',
													 list(self.agent.qf1_target.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf1_target.parameters())[0].grad is not None:
						param_list = list(self.agent.qf1_target.parameters())
						self.tb_logger.writer.add_scalar('gradients/q1_target',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q2_network',
													 list(self.agent.qf2.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf2.parameters())[0].grad is not None:
						param_list = list(self.agent.qf2.parameters())
						self.tb_logger.writer.add_scalar('gradients/q2_network',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q2_target',
													 list(self.agent.qf2_target.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf2_target.parameters())[0].grad is not None:
						param_list = list(self.agent.qf2_target.parameters())
						self.tb_logger.writer.add_scalar('gradients/q2_target',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/policy',
													 list(self.agent.policy.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.policy.parameters())[0].grad is not None:
						param_list = list(self.agent.policy.parameters())
						self.tb_logger.writer.add_scalar('gradients/policy',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)

			print("Iteration -- {}, Success rate -- {:.3f}, Avg. return -- {:.3f}, Success rate online -- {:.3f}, Avg. return online -- {:.3f}, " \
			"Success rate train -- {:.3f}, Avg. return train -- {:.3f}, Elapsed time {:5d}[s]"
					.format(iteration, np.mean(success_rate), np.mean(returns), np.mean(online_success_rate), np.mean(online_returns),
					np.mean(success_rate_train), np.mean(returns_train), 
							int(time.time() - self._start_time)), train_stats)

	def sample_rl_batch(self, tasks, batch_size):
		''' sample batch of unordered rl training data from a list/array of tasks '''
		# this batch consists of transitions sampled randomly from replay buffer
		batches = [ptu.np_to_pytorch_batch(
			self.storage.random_batch(task, batch_size)) for task in tasks]
		unpacked = [utl.unpack_batch(batch) for batch in batches]
		# group elements together
		unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
		unpacked = [torch.cat(x, dim=0) for x in unpacked]
		return unpacked
	
	def sample_random_context_batch(self, tasks, context_size, trainset=True):
		if trainset:
			storage = self.storage
		else:
			storage = self.eval_storage
		batches = [ptu.np_to_pytorch_batch(
			storage.sample_random_context(task, context_size)) for task in tasks]
		unpacked = [utl.unpack_batch(batch) for batch in batches]
		# group elements together
		unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
		unpacked = [torch.cat(x, dim=0) for x in unpacked]
		return unpacked



	def _start_training(self):
		self._n_rl_update_steps_total = 0
		self._start_time = time.time()

	def training_mode(self, mode):
		self.agent.train(mode)
		self.world_model.train(mode)

	def get_task_embeddings(self, indices):
		obs_context, actions_context, rewards_context, next_obs_context, terms_context, pre_rewards_context = self.sample_random_context_batch(indices, self.context_size, trainset=False)
		task_desc = self.world_model.context_encoding(states=obs_context, actions=actions_context, pre_rewards=pre_rewards_context)
		task_desc = ptu.get_numpy(task_desc)
		return task_desc
	

def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--env-type', default='gridworld')
	parser.add_argument('--env-type', default='point_robot_sparse')
	# parser.add_argument('--env-type', default='cheetah_vel')
	# parser.add_argument('--env-type', default='cheetah_vel_sparse')
	# parser.add_argument('--env-type', default='ant_dir')
	# parser.add_argument('--env-type', default='ant_semicircle')
	# parser.add_argument('--env-type', default='cheetah_dir')
	# parser.add_argument('--env-type', default='hopper_param')
	# parser.add_argument('--env-type', default='hopper_param_sparse')
	# parser.add_argument('--env-type', default='walker_param')
	# parser.add_argument('--env-type', default='walker_param_sparse')
	# parser.add_argument('--env-type', default='reach')
	# parser.add_argument('--env-type', default='grid_block')
	args, rest_args = parser.parse_known_args()
	env = args.env_type

	if env == 'cheetah_vel':
		args = args_cheetah_vel.get_args(rest_args)
	elif env == 'cheetah_vel_sparse':
		args = args_cheetah_vel_sparse.get_args(rest_args)
	elif env == 'cheetah_dir':
		args = args_cheetah_dir.get_args(rest_args)
	elif env == 'ant_dir':
		args = args_ant_dir.get_args(rest_args)
	elif env == 'ant_semicircle':
		args = args_ant_semicircle.get_args(rest_args)
	elif env == 'hopper_param':
		args = args_hopper_param.get_args(rest_args)
	elif env == 'hopper_param_sparse':
		args = args_hopper_param_sparse.get_args(rest_args)
	elif env == 'walker_param':
		args = args_walker_param.get_args(rest_args)
	elif env == 'walker_param_sparse':
		args = args_walker_param_sparse.get_args(rest_args)
	elif env == 'point_robot_sparse':
		args = args_point_robot_sparse.get_args(rest_args)
	else:
		raise NotImplementedError

	print(args.use_gpu)
	set_gpu_mode(torch.cuda.is_available() and args.use_gpu, gpu_id=0)
	print(ptu.device)

	args, _ = off_utl.expand_args(args) # add env information to args
	#print(args)


	dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
	assert args.num_train_tasks + args.num_eval_tasks == len(goals)
	train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
	eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]
 

	learner = MetaSTAR(args, train_dataset, train_goals, eval_dataset, eval_goals)
 
	learner.train()


if __name__ == '__main__':
	main()
