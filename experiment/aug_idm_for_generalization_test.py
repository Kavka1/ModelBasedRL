from copy import copy
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import yaml
from torch.utils.tensorboard import SummaryWriter

from ModelBasedRL.agent.utils import check_path, array2tensor
from ModelBasedRL.agent.sac import SAC
from ModelBasedRL.agent.utils import soft_update
from ModelBasedRL.buffer.trans_buffer import Buffer
from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM



def action_augmentation(action_batch: torch.tensor, delta: float, noise_range: List[float]) -> np.array:
    batch_size  = action_batch.shape[0]
    action_dim  = action_batch.shape[1]

    noise_batch_for_a = np.random.uniform(
        noise_range[0],
        noise_range[1],
        (batch_size, action_dim),
    ) * delta
    noise_batch_for_a = torch.from_numpy(noise_batch_for_a).to(action_batch.device).float()

    augmented_action_batch = copy(action_batch) + noise_batch_for_a
    return augmented_action_batch


class IDM_learner_under_augmentation(object):
    def __init__(self, config: Dict, delta: float, noise_range: List[float]) -> None:
        self.lr = config['lr']
        self.device = torch.device(config['device'])
        self.batch_size = config['batch_size_model']
        self.delta = delta
        self.noise_range = noise_range

        self.inverse_dynamics_model = DiagGaussianIDM(
            o_dim= config['model_config']['o_dim'],
            a_dim= config['model_config']['a_dim'],
            hidden_layers= config['model_config']['dynamics_hidden_layers'],
            logstd_min= config['model_config']['model_logstd_min'],
            logstd_max= config['model_config']['model_logstd_max']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.inverse_dynamics_model.parameters(), self.lr, weight_decay=0.01)

    def train(self, buffer: Buffer) -> float:
        if len(buffer) < self.batch_size:
            return 0.

        obs, a, r, done, obs_ = buffer.sample(self.batch_size)
        # augmentation the batch transferring the a to a + delta * Uniform[a,b]
        a = action_augmentation(a, self.delta, self.noise_range)            

        obs, a, r, done, obs_ = array2tensor(obs, a, r, done, obs_, self.device)
        return self.train_with_batch(obs, a, obs_)

    def train_with_batch(self, obs: torch.tensor, a: torch.tensor, obs_: torch.tensor) -> float:
        # augmentation the batch transferring the a to a + delta * Uniform[a,b]
        a = action_augmentation(a, self.delta, self.noise_range)            

        dist = self.inverse_dynamics_model(obs, obs_)
        loss = - dist.log_prob(a).mean()
        loss += 0.01 * (self.inverse_dynamics_model.logstd_max.sum() - self.inverse_dynamics_model.logstd_min.sum())    # penalty for extreme logstd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, path: str, remark: str) -> None:
        check_path(path)
        torch.save(self.inverse_dynamics_model.state_dict(), path+f'IDM_{remark}')
        print(f"| - Model of IDM saved to {path} - |")



class SAC_with_multiple_IDM(SAC):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)

    def train_ac_and_idms(self, buffer: Buffer, all_idms: List[IDM_learner_under_augmentation]) -> Dict:
        if len(buffer) < self.batch_size:
            log = {
                'loss_q': 0, 
                'loss_policy': 0, 
                'loss_alpha': 0, 
                'alpha': self.logger_alpha
            }
            for _ in range(len(all_idms)):
                log.update({f'loss_idm_{_}': 0.})
            return log

        obs, a, r, done, obs_ = buffer.sample(self.batch_size)
        obs, a, r, done, obs_ = array2tensor(obs, a, r, done, obs_, self.device)

        idm_losses = []
        for idm in all_idms:
            idm_losses.append(idm.train_with_batch(obs, a, obs_))

        with torch.no_grad():
            next_a, next_a_logprob, dist = self.policy(obs_)
            next_q1_tar, next_q2_tar = self.value_tar(obs_, next_a)
            next_q_tar = torch.min(next_q1_tar, next_q2_tar)
            q_update_tar = r + (1 - done) * self.gamma * (next_q_tar - self.alpha * next_a_logprob)
        q1_pred, q2_pred = self.value(obs, a)
        loss_q = F.mse_loss(q1_pred, q_update_tar) + F.mse_loss(q2_pred, q_update_tar)
        self.optimizer_value.zero_grad()
        loss_q.backward(retain_graph=True)
        self.optimizer_value.step()

        self.logger_loss_q = loss_q.item()
        self.update_count += 1

        if self.update_count % self.train_policy_delay == 0:
            a_new, a_new_logprob, dist_new = self.policy(obs)
            loss_policy = (self.alpha * a_new_logprob - self.value.call_Q1(obs, a_new)).mean()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.optimizer_policy.step()

            a_new_logprob = torch.tensor(a_new_logprob.tolist(), requires_grad=False, device=self.device)
            loss_alpha = (- torch.exp(self.log_alpha) * (a_new_logprob + self.target_entropy)).mean()
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()

            self.alpha = torch.exp(self.log_alpha)

            self.logger_alpha = self.alpha.item()
            self.logger_loss_alpha = loss_alpha.item()
            self.logger_loss_policy = loss_policy.item()
            
        soft_update(self.value, self.value_tar, self.tau)

        log = {
            'loss_q': self.logger_loss_q, 
            'loss_policy': self.logger_loss_policy, 
            'loss_alpha': self.logger_loss_alpha,
            'alpha': self.logger_alpha
        }
        for i in range(len(all_idms)):
            log.update({f'loss_idm_{i}': idm_losses[i]})
        return log



def main(config: Dict, exp_name: str = ''):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    # make experiment file dir
    if exp_name is not '':
        exp_file = f'{exp_name}-' + f"Halfcheetah-{config['seed']}"
    else:
        exp_file = f"Halfcheetah-{config['seed']}"
    exp_path = config['result_path'] + exp_file
    while os.path.exists(exp_path):
        exp_path += '_*'
    config.update({'exp_path': exp_path + '/'})
    check_path(config['exp_path'])

    logger = SummaryWriter(config['exp_path'])

    env = gym.make('HalfCheetah-v2')
    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0]
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)

    agent = SAC_with_multiple_IDM(config)
    buffer = Buffer(config['buffer_size'])

    all_idm_learners = []
    for delta in config['deltas']:
        all_idm_learners.append(
            IDM_learner_under_augmentation(config, delta, config['noise_range'])
        )    

    total_step, total_episode = 0, 0
    best_score = 0
    all_idm_best_acc = [1 for _ in range(len(all_idm_learners))]

    obs = env.reset()
    while total_step <= config['max_timesteps']:
        action = agent.choose_action(obs, True)
        next_obs, reward, done, info = env.step(action)
        buffer.save_trans((obs, action, reward, done, next_obs))
        loss_dict = agent.train_ac_and_idms(buffer, all_idm_learners)

        if done:
            total_episode += 1
            obs = env.reset()
        else:
            obs = next_obs

        if total_step % config['eval_interval'] == 0:
            eval_score = agent.evaluate(env, config['eval_episode'])
            if eval_score > best_score:
                agent.save_policy(config['exp_path'], 'best')
                best_score = eval_score

            print(f"| Step: {total_step} | Episode: {total_episode} | Eval_Return: {eval_score} | Loss: {loss_dict} |")
            logger.add_scalar('Eval/Eval_Return', eval_score, total_step)
            for loss_name, loss_value in list(loss_dict.items()):
                logger.add_scalar(f'Train/{loss_name}', loss_value, total_step)
        
        if total_step % config['save_interval'] == 0:
            agent.save_policy(config['exp_path'], f'{total_step}')
            for i, idm_learner in enumerate(all_idm_learners):
                idm_learner.save_model(config['exp_path'], f'{i}_{total_step}')
                if all_idm_best_acc[i] > loss_dict[f'loss_idm_{i}']:
                    idm_learner.save_model(config['exp_path'], f'{i}_best')
                    all_idm_best_acc[i] = loss_dict[f'loss_idm_{i}']

        total_step += 1

    # save the final models
    agent.save_policy(config['exp_path'], 'final')
    for i, idm_learner in enumerate(all_idm_learners):
        idm_learner.save_model(config['exp_path'], f'{i}_final')



if __name__ == '__main__':
    config = {
        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'policy_hidden_layers': [128, 128],
            'value_hidden_layers': [128, 128],
            'dynamics_hidden_layers': [256, 256],
            'policy_logstd_min': -20,
            'policy_logstd_max': 2,
            'model_logstd_min': -10,
            'model_logstd_max': 0.5,
        },
        'deltas': [0.2, 0.4, 0.6, 0.8],
        'noise_range': [-0.1, 0.1],

        'seed': 10,
        'buffer_size': 1000000,
        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 256,
        'batch_size_model': 256,
        'initial_alpha': 1,
        'train_policy_delay': 2,
        'device': 'cpu',
        'max_timesteps': 1000000,
        'eval_interval': 2000,
        'save_interval': 100000,
        'eval_episode': 5,
        'result_path': '/home/xukang/GitRepo/ModelBasedRL/results/aug_idm_test/'
    }

    main(config, '')