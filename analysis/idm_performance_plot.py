from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
import dmc2gym
import matplotlib.pyplot as plt
import seaborn as sns

from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM


def load_model_and_config(path: str, remark: str) -> Tuple[nn.Module, Dict]:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_config = config['model_config']
    
    idm_model = DiagGaussianIDM(
        model_config['o_dim'],
        model_config['a_dim'],
        model_config['dynamics_hidden_layers'],
        model_config['logstd_min'],
        model_config['logstd_max']
    )
    idm_model.load_state_dict(torch.load(path + f'IDM_{remark}'))
    return idm_model, config


def main(path: str, remark: str, collect_action_num: int = 16) -> None:
    idm_model, config = load_model_and_config(path, remark)

    env = dmc2gym.make(
        domain_name=config['domain_name'],
        task_name=config['task_name'],
        seed=config['seed'],
        visualize_reward=False,
        from_pixels=False
    )

    trans_seq = []
    done = False
    obs = env.reset()
    while len(trans_seq) < collect_action_num:
        a = env.action_space.sample()
        obs_, r, done, _ = env.step(a)
        if done:
            obs = env.reset()
        if np.random.uniform() > 0.5:
            trans_seq.append((obs, a, obs_))
    
    action_dist_seq = []
    for trans in trans_seq:
        obs_tensor = torch.from_numpy(trans[0]).float()
        next_obs_tensor = torch.from_numpy(trans[-1]).float()
        a_pred_dict = idm_model(obs_tensor, next_obs_tensor)
        action_dist_seq.append(a_pred_dict)

    sns.set_theme(style='white')
    fig, axs = plt.subplots(nrows=4, ncols=4, tight_layout=True)
    for i, ax in enumerate(axs):
        dist = action_dist_seq[i]
        action_samples = dist.sample_n(1000)
        action_samples = action_samples.detach().item()
        df = pd.DataFrame({
            'action_dim_0': [a[0] for a in action_samples],
            'action_dim_1': [a[1] for a in action_samples] 
        })
        sns.jointplot(
            data= df,
            x= 'action_dim_0',
            y= 'action_dim_1',
            kind='kde',
            ax= ax
        )

    plt.show()


if __name__ == '__main__':
    main(
        path= '/home/xukang/GitRepo/ModelBasedRL/results/',
        remark= 'best',
        collect_action_num=16
    )