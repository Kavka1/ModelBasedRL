from dis import dis
from typing import List, Tuple, Dict, Union
import numpy as np
import torch
import yaml
import datetime
import dmc2gym
from torch.utils.tensorboard import SummaryWriter

from ModelBasedRL.agent.utils import check_path, array2tensor
from ModelBasedRL.agent.sac import SAC
from ModelBasedRL.buffer.trans_buffer import Buffer
from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM


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


def main(config: Dict):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    config.update({
        'exp_path': config['result_path'] + f"{config['domain_name']}_{config['task_name']}-{config['seed']}-{datetime.datetime.now().strftime('%m-%d_%H-%M')}/"
    })
    check_path(config['exp_path'])
    logger = SummaryWriter(config['exp_path'])

    env = dmc2gym.make(
        domain_name= config['domain_name'],
        task_name= config['task_name'],
        seed= config['seed'],
        visualize_reward= False,
        from_pixels= False,
    )
    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0]
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)

    agent = SAC(config)
    buffer = Buffer(config['buffer_size'])
    inverse_model_learner = IDM_learner(config)

    total_step, total_episode = 0, 0
    best_score, best_accuracy = 0, 0
    obs = env.reset()
    while total_step < config['max_timesteps']:
        action = agent.choose_action(obs, True)
        next_obs, reward, done, info = env.step(action)
        buffer.save_trans((obs, action, reward, done, next_obs))
        
        loss_dict = agent.train_ac(buffer)
        loss_idm = inverse_model_learner.train(buffer)
        loss_dict.update({'loss_IDM': loss_idm})

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
            if best_accuracy > loss_idm:
                inverse_model_learner.save_model(config['exp_path'], 'best')
                best_accuracy = loss_idm

        total_step += 1

if __name__ == '__main__':
    config = {
        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'policy_hidden_layers': [256, 256],
            'value_hidden_layers': [256, 256],
            'dynamics_hidden_layers': [256, 256],
            'policy_logstd_min': -20,
            'policy_logstd_max': 2,
            'model_logstd_min': -10,
            'model_logstd_max': 0.5,
        },
        'domain_name': 'reacher',
        'task_name': 'easy',
        'seed': 10,
        'buffer_size': 500000,
        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.001,
        'batch_size': 256,
        'batch_size_model': 2000,
        'initial_alpha': 1,
        'train_policy_delay': 2,
        'device': 'cpu',
        'max_timesteps': 500000,
        'eval_interval': 10000,
        'save_interval': 50000,
        'eval_episode': 10,
        'result_path': '/home/xukang/GitRepo/ModelBasedRL/results/dmc-sac-idm_test/'
    }

    for domain_task in [
        ('reacher', 'easy'),
        #('cheetah', 'run'),
        #('walker', 'walk')
    ]:
        config.update({
            'domain_name': domain_task[0],
            'task_name': domain_task[1]
        })
        main(config)