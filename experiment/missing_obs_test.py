from dis import dis
from typing import List, Tuple, Dict, Union
import numpy as np
import os
import torch
import yaml
import datetime
from torch.utils.tensorboard import SummaryWriter

from ModelBasedRL.env.lack_obs_wrap.walker import Missing_Joint_Vel_Walker

from ModelBasedRL.agent.utils import check_path, array2tensor
from ModelBasedRL.agent.sac import SAC
from ModelBasedRL.model.policy.flat_policy import DiagGaussianPolicy
from ModelBasedRL.buffer.trans_buffer import Buffer
from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM
from ModelBasedRL.model.dynamics.forward_dynamics import DiagGaussianFDM


class IDM_learner(object):
    def __init__(self, config: Dict) -> None:
        self.lr = config['lr']
        self.device = torch.device(config['device'])
        self.batch_size = config['batch_size_model']

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
        obs, a, r, done, obs_ = array2tensor(obs, a, r, done, obs_, self.device)

        return self.train_with_batch(obs, a, obs_)

    def train_with_batch(self, obs: torch.tensor, a: torch.tensor, obs_: torch.tensor) -> float:
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



class FDM_learner(object):
    def __init__(self, config: Dict) -> None:
        self.lr = config['lr']
        self.device = torch.device(config['device'])
        self.batch_size = config['batch_size_model']

        self.forward_dynamics_model = DiagGaussianFDM(
            o_dim= config['model_config']['o_dim'],
            a_dim= config['model_config']['a_dim'],
            hidden_layers= config['model_config']['dynamics_hidden_layers'],
            logstd_min= config['model_config']['model_logstd_min'],
            logstd_max= config['model_config']['model_logstd_max']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.forward_dynamics_model.parameters(), self.lr, weight_decay=0.01)

    def train(self, buffer: Buffer) -> float:
        if len(buffer) < self.batch_size:
            return 0.

        obs, a, r, done, obs_ = buffer.sample(self.batch_size)
        obs, a, r, done, obs_ = array2tensor(obs, a, r, done, obs_, self.device)

        return self.train_with_batch(obs, a, obs_)

    def train_with_batch(self, obs: torch.tensor, a: torch.tensor, obs_: torch.tensor) -> float:
        dist = self.forward_dynamics_model(obs, a)
        loss = - dist.log_prob(obs_).mean()
        loss += 0.01 * (self.forward_dynamics_model.logstd_max.sum() - self.forward_dynamics_model.logstd_min.sum())    # penalty for extreme logstd
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, path: str, remark: str) -> None:
        check_path(path)
        torch.save(self.forward_dynamics_model.state_dict(), path+f'FDM_{remark}')
        print(f"| - Model of FDM saved to {path} - |")



def main(config: Dict, exp_name: str = ''):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    # make experiment file dir
    if exp_name is not '':
        exp_file = f'{exp_name}-' + f"Walker-missing_{config['missing_joint']}-{config['seed']}"
    else:
        exp_file = f"Walker-missing_{config['missing_joint']}-{config['seed']}"
    exp_path = config['result_path'] + exp_file
    while os.path.exists(exp_path):
        exp_path += '_*'
    config.update({'exp_path': exp_path + '/'})
    check_path(config['exp_path'])

    logger = SummaryWriter(config['exp_path'])

    env = Missing_Joint_Vel_Walker(config['missing_joint'])
    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0]
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)

    agent = SAC(config)
    buffer = Buffer(config['buffer_size'])
    inverse_model_learner = IDM_learner(config)
    forward_model_learner = FDM_learner(config)

    total_step, total_episode = 0, 0
    best_score, best_accuracy_idm, best_accuracy_fdm = 0, 1, 1
    obs = env.reset()
    while total_step <= config['max_timesteps']:
        action = agent.choose_action(obs, True)
        next_obs, reward, done, info = env.step(action)
        buffer.save_trans((obs, action, reward, done, next_obs))
        
        loss_dict = agent.train_ac_and_models(buffer, inverse_model_learner, forward_model_learner)

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
            inverse_model_learner.save_model(config['exp_path'], f'{total_step}')
            forward_model_learner.save_model(config['exp_path'], f'{total_step}')
            if best_accuracy_idm > loss_dict['loss_IDM']:
                inverse_model_learner.save_model(config['exp_path'], 'best')
                best_accuracy_idm = loss_dict['loss_IDM']
            if best_accuracy_fdm > loss_dict['loss_FDM']:
                forward_model_learner.save_model(config['exp_path'], 'best')
                best_accuracy_fdm = loss_dict['loss_FDM']

        total_step += 1


def demo(path: str, remark: str) -> None:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    policy = DiagGaussianPolicy(
        config['model_config']['o_dim'],
        config['model_config']['a_dim'],
        config['model_config']['policy_hidden_layers'],
        config['model_config']['policy_logstd_min'],
        config['model_config']['policy_logstd_max'],
    )
    policy.load_model(path + f'policy_{remark}')
    env = Missing_Joint_Vel_Walker(config['missing_joint'])

    for _ in range(100):
        done = False
        episode_r = 0
        obs = env.reset()
        while not done:
            env.render()
            obs = torch.from_numpy(obs).float()
            a = policy.act(obs, False)
            obs, r, done, info = env.step(a)
            episode_r += r
        print(f"Episode {_} Episode Reward {episode_r}")




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
        'missing_joint': ['thigh', 'leg', 'thigh'],
        'seed': 10,
        'buffer_size': 800000,
        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.001,
        'batch_size': 256,
        'batch_size_model': 256,
        'initial_alpha': 1,
        'train_policy_delay': 2,
        'device': 'cuda',
        'max_timesteps': 800000,
        'eval_interval': 20000,
        'save_interval': 100000,
        'eval_episode': 5,
        'result_path': '/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/'
    }

    for missing_jnt in [
        [],
        ['foot'],
        ['foot', 'leg'],
        ['foot', 'leg', 'thigh']
    ]:
        config.update({
            'missing_joint': missing_jnt
        })
        main(config, 'both_model')

    #demo("/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/Walker_missing_['foot', 'leg', 'thigh']-10/", 'best')