from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
import gym
import matplotlib.pyplot as plt
import seaborn as sns

from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM
from ModelBasedRL.env.dynamics_variance_wrap.Halfcheetah import HalfCheetah_Mass_Variance_Wrapper


def load_config(path: str) -> Dict:
    with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(path: str, config: Dict, model_idx: int, remark: str) -> nn.Module:
    model_config = config['model_config']
    
    idm_model = DiagGaussianIDM(
        model_config['o_dim'],
        model_config['a_dim'],
        model_config['dynamics_hidden_layers'],
        model_config['model_logstd_min'],
        model_config['model_logstd_max']
    )
    idm_model.load_state_dict(torch.load(path + f'IDM_{model_idx}_{remark}'))
    return idm_model


def main(path: str, remark: str, num_models: int = 4, collect_action_num: int = 2000) -> None:
    config = load_config(path)
    all_idms = [load_model(path, config, i, remark) for i in range(num_models)]
    env = HalfCheetah_Mass_Variance_Wrapper(
        gym.make('HalfCheetah-v2')
    )

    fricition_shifts = [round(item * 0.1, 1) for item in list(range(5, 20))]
    df_for_action_index = [[] for _ in range(env.action_space.shape[0])]

    for fric_shift_mag in fricition_shifts:
        env.reset_friction(fric_shift_mag)
        # collect transitions
        obs_seq, a_seq, next_obs_seq = [], [], []
        done = False
        obs = env.reset()
        while len(obs_seq) < collect_action_num:
            a = env.action_space.sample()
            obs_, r, done, _ = env.step(a)
            if done:
                obs = env.reset()
            if np.random.uniform() > 0.5:
                obs_seq.append(obs)
                a_seq.append(a)
                next_obs_seq.append(obs_)

        obs_tensor      = torch.from_numpy(np.stack(obs_seq, 0)).float()
        next_obs_tensor = torch.from_numpy(np.stack(next_obs_seq, 0)).float()
        a_tensor        = torch.from_numpy(np.stack(a_seq, 0)).float()

        for i, idm in enumerate(all_idms):
            a_pred_dict = idm(obs_tensor, next_obs_tensor)
            a_pred_accu = a_pred_dict.log_prob(a_tensor).mean(0).detach().numpy().tolist()
            for a_id in range(len(df_for_action_index)):
                df_for_action_index[a_id].append(pd.DataFrame({
                                                    'Mag':          [fric_shift_mag],
                                                    'Model':        [i],
                                                    'Acc':          [a_pred_accu[a_id]]
                                                }))

    for a_id in range(len(df_for_action_index)):
        df_for_action_index[a_id] = pd.concat(df_for_action_index[a_id])

    sns.set_theme(style='whitegrid')
    fig, axs = plt.subplots(nrows=len(df_for_action_index), ncols=1, tight_layout=True, figsize=(10, 15))
    for i, ax in enumerate(axs):
        sns.barplot(
            data=df_for_action_index[i],
            x = 'Mag',
            y = 'Acc',
            hue = 'Model',
            ax = ax
        )
        ax.set_title(f'HalfCheetah - IDM Prediction Accuracy under various foot mass shifts', fontsize=12)
        ax.legend().set_title('')

    plt.savefig('/data/xukang/Project/ModelBasedRL/analysis/aug_idm_generalization/fig.png')
    #plt.show()


if __name__ == '__main__':
    main(
        path= '/data/xukang/Project/ModelBasedRL/results/aug_idm_test/Halfcheetah-10/',
        remark= 'best',
        num_models=4,
        collect_action_num=2000
    )