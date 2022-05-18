from typing import Dict, List, Tuple
import gym
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from ModelBasedRL.env.lack_obs_wrap.walker import Missing_Joint_Vel_Walker
from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM


def load_model(exp_path: str, model_config: Dict, remark: str) -> Tuple[nn.Module, Dict]:
    idm_model = DiagGaussianIDM(
        model_config['o_dim'],
        model_config['a_dim'],
        model_config['dynamics_hidden_layers'],
        model_config['model_logstd_min'],
        model_config['model_logstd_max']
    )
    idm_model.load_state_dict(torch.load(exp_path + f'IDM_{remark}'))
    return idm_model


def plot_errorbar(all_path: List[str], remark: str) -> None:
    all_models = []
    all_envs = []
    for path in all_path:
        with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        env = Missing_Joint_Vel_Walker(config['missing_joint'])
        model = load_model(path, config['model_config'], remark)
        
        all_envs.append(env)
        all_models.append(model)

    original_env = Missing_Joint_Vel_Walker([])
    dim_a = original_env.action_space.shape[0]
    collect_action_num = dim_a * 4

    trans_seq = []
    done = False
    obs = original_env.reset()
    while len(trans_seq) < collect_action_num:
        a = env.action_space.sample()
        obs_, r, done, _ = original_env.step(a)
        if done:
            obs = original_env.reset()
        if np.random.uniform() > 0.5:
            trans_seq.append((obs, a, obs_))
    
    action_seq_dist = {f'model_{i}': [] for i in range(len(all_models))}
    for trans in trans_seq:
        for i in range(len(all_models)):
            model, env = all_models[i], all_envs[i]

            obs, next_obs = trans[0], trans[-1]
            processed_obs, processed_next_obs = env._process_obs(obs), env._process_obs(next_obs)

            obs_tensor = torch.from_numpy(processed_obs).float()
            next_obs_tensor = torch.from_numpy(processed_next_obs).float()
            a_pred_dict = model(obs_tensor, next_obs_tensor)
            action_seq_dist[f'model_{i}'].append(a_pred_dict)


    sns.set_theme(style='whitegrid')
    fig, axs = plt.subplots(nrows=dim_a-1, ncols=4, tight_layout=True, figsize=(15, 10))
    for i in range(dim_a-1):
        for j in range(4):
            ax = axs[i,j]
            df = []

            action_1_mean, action_2_mean = [], []
            action_1_upper, action_1_lower, action_2_upper, action_2_lower = [], [], [], []

            for k in range(len(all_models)):
                dist = action_seq_dist[f'model_{k}'][i*4+j]
                action_samples = dist.sample_n(1000)
                action_samples = action_samples.detach().numpy().tolist()
                
                action_samples_1 = [a[i] for a in action_samples]
                action_samples_2 = [a[i+1] for a in action_samples]
                df.append(
                    pd.DataFrame({
                        f'action_dim_{i}':      action_samples_1,     #+ [trans_seq[i*4+j][1][0]]*1000,
                        f'action_dim_{i+1}':    action_samples_2,   #+ [trans_seq[i*4+j][1][1]]*1000,
                        'dist': [f'model_{k}'] * len(action_samples)                #+ ['true'] * 1000
                    })
                )

                mean_1, mean_2 = np.mean(action_samples_1), np.mean(action_samples_2)
                std_1, std_2 = np.std(action_samples_1), np.std(action_samples_2)

                ax.errorbar(
                    x= [mean_1],
                    y= [mean_2],
                    xerr=[[std_1], [std_1]],
                    yerr=[[std_2], [std_2]],
                    fmt='o',
                    label=f'model_{k}'
                )
            #ax.legend([f'model_{l}' for l in range(len(all_models))])

            ax.plot([trans_seq[i*4+j][1][0]], [trans_seq[i*4+j][1][1]], '*', color='black')
            #ax.set_xlim([-1,1])
            #ax.set_ylim([-1,1])
            if i==0 and j==0:
                pass
                #ax.legend().set_title('')
            else:
                ax.legend().remove()
            #ax.set_title(f'true action: {trans_seq[i*4+j][1]}', fontsize=10)

    plt.show()


def plot_pred_var_difference(all_path: List[str], remark: str) -> None:
    all_models = []
    all_envs = []
    for path in all_path:
        with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        env = Missing_Joint_Vel_Walker(config['missing_joint'])
        model = load_model(path, config['model_config'], remark)
        
        all_envs.append(env)
        all_models.append(model)

    original_env = Missing_Joint_Vel_Walker([])
    dim_a = original_env.action_space.shape[0]
    collect_action_num = 1000

    trans_seq = []
    done = False
    obs = original_env.reset()
    while len(trans_seq) < collect_action_num:
        a = env.action_space.sample()
        obs_, r, done, _ = original_env.step(a)
        if done:
            obs = original_env.reset()
        if np.random.uniform() > 0.5:
            trans_seq.append((obs, a, obs_))
    
    action_seq_dist = {f'model_{i}': [] for i in range(len(all_models))}
    for trans in trans_seq:
        for i in range(len(all_models)):
            model, env = all_models[i], all_envs[i]

            obs, next_obs = trans[0], trans[-1]
            processed_obs, processed_next_obs = env._process_obs(obs), env._process_obs(next_obs)

            obs_tensor = torch.from_numpy(processed_obs).float()
            next_obs_tensor = torch.from_numpy(processed_next_obs).float()
            a_pred_dict = model(obs_tensor, next_obs_tensor)
            action_seq_dist[f'model_{i}'].append(a_pred_dict)
    
    all_labels = [
        'missing none obs info',
        'missing foot vel info',
        'missing foot&leg vel info',
        'missing foot&leg&thigh vel info'
    ]
    model2missingobs = {
        f'model_{k}': all_labels[k] for k in range(len(all_models))
    }

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1 , 1)
    df = []
    for k in range(len(all_models)):
        model = all_models[k]

        a_pred_vars = [dist.variance.detach().tolist() for dist in action_seq_dist[f'model_{k}']]
        
        for l in range(dim_a):
            df.append(pd.DataFrame({
                'action prediction variance': [a_pred_var[l] for a_pred_var in a_pred_vars],
                'action dim': [f'dim_{l}'] * len(a_pred_vars),
                'model': [model2missingobs[f'model_{k}']] * len(a_pred_vars)
            }))
    
    df = pd.concat(df)
    sns.barplot(
        data=df,
        x='action dim',
        y='action prediction variance',
        hue= 'model',
        ax=ax,
        ci='sd',
    )

    plt.legend().set_title('')
    ax.set_title('Action prediction variance of Inverse Dynamics Model trained in difference observation setting', fontsize=10)
    ax.set_xlabel('Action Dimension', fontsize=10)
    ax.set_ylabel('Action Prediction Variance of Inverse Dynamics Model', fontsize=10)
    plt.show()


if __name__ == '__main__':
    '''
    plot_errorbar(
        all_path= [
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/Walker_missing_[]-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/Walker_missing_['foot']-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/Walker_missing_['foot', 'leg']-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/Walker_missing_['foot', 'leg', 'thigh']-10/",
        ],
        remark= 'best',
    )
    '''
    plot_pred_var_difference(
        all_path= [
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_[]-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_['foot']-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_['foot', 'leg']-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_['foot', 'leg', 'thigh']-10/",
        ],
        remark= '800000',
    )