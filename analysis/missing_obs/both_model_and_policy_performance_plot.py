from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import seaborn as sns
import torch
import yaml
from cycler import cycler
import torch.nn as nn

from ModelBasedRL.model.dynamics.inverse_dynamics import DiagGaussianIDM
from ModelBasedRL.model.dynamics.forward_dynamics import DiagGaussianFDM
from ModelBasedRL.model.policy.flat_policy import DiagGaussianPolicy
from ModelBasedRL.env.lack_obs_wrap.walker import Missing_Joint_Vel_Walker


plt.rcParams['axes.prop_cycle']  = cycler(color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2','#59A14E',
                                                 '#EDC949','#B07AA2','#FF9DA7','#9C755F','#BAB0AC'])


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    xdata = l.get_xdata()
                    
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
                    elif len(xdata) == 2 and xdata[0]> xmin and xdata[1] < xmax and xdata[0]!=xdata[1]:
                        l.set_xdata([xmid-fac*0.6*xhalf, xmid+fac*0.6*xhalf])


def load_both_model(exp_path: str, model_config: Dict, remark: str) -> Tuple[nn.Module, nn.Module]:
    idm_model = DiagGaussianIDM(
        model_config['o_dim'],
        model_config['a_dim'],
        model_config['dynamics_hidden_layers'],
        model_config['model_logstd_min'],
        model_config['model_logstd_max']
    )
    idm_model.load_state_dict(torch.load(exp_path + f'IDM_{remark}'))

    fdm_model = DiagGaussianFDM(
        model_config['o_dim'],
        model_config['a_dim'],
        model_config['dynamics_hidden_layers'],
        model_config['model_logstd_min'],
        model_config['model_logstd_max']
    )
    fdm_model.load_state_dict(torch.load(exp_path + f'FDM_{remark}'))

    return idm_model, fdm_model


def load_policy(exp_path: str, model_config: Dict, remark: str) -> nn.Module:
    policy = DiagGaussianPolicy(
        model_config['o_dim'],
        model_config['a_dim'],
        model_config['policy_hidden_layers'],
        model_config['policy_logstd_min'],
        model_config['policy_logstd_max']
    )
    policy.load_model(exp_path + f'policy_{remark}')
    
    return policy


def rollout(policy: DiagGaussianPolicy, env: Missing_Joint_Vel_Walker, num_episode: int = 50) -> List[float]:
    all_episode_r = []
    for _ in range(num_episode):
        obs = env.reset()
        done = False
        episode_r = 0
        while not done:
            obs = torch.from_numpy(obs).float()
            a = policy.act(obs, False).detach().numpy()
            obs, r, done, info = env.step(a)
            episode_r += r
        all_episode_r.append(episode_r)
    return all_episode_r


def performance_and_model_variance_plot(all_path: List[str], remark: str) -> None:
    all_policy, all_fdm, all_idm = [], [], []
    all_env = []
    for path in all_path:
        with open(path + 'config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        env = Missing_Joint_Vel_Walker(config['missing_joint'])
        idm, fdm = load_both_model(path, config['model_config'], remark)
        policy = load_policy(path, config['model_config'], remark)

        all_env.append(env)
        all_fdm.append(fdm)
        all_idm.append(idm)
        all_policy.append(policy)

    original_env = Missing_Joint_Vel_Walker([])
    collect_action_num = 2000

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
    
    idm_action_seq_dist = {f'idm_{i}': [] for i in range(len(all_idm))}
    fdm_obs_seq_dist = {f'fdm_{i}': [] for i in range(len(all_fdm))}

    for trans in trans_seq:
        for i in range(len(all_fdm)):
            fdm, idm, env = all_fdm[i], all_idm[i], all_env[i]

            obs, a, next_obs = trans[0], trans[1], trans[-1]
            processed_obs, processed_next_obs = env._process_obs(obs), env._process_obs(next_obs)

            obs_tensor = torch.from_numpy(processed_obs).float()
            a_tensor = torch.from_numpy(a).float()
            next_obs_tensor = torch.from_numpy(processed_next_obs).float()

            a_pred_dict = idm(obs_tensor, next_obs_tensor)
            obs_pred_dict = fdm(obs_tensor, a_tensor)

            idm_action_seq_dist[f'idm_{i}'].append(a_pred_dict)
            fdm_obs_seq_dist[f'fdm_{i}'].append(obs_pred_dict)
    
    all_labels = [
        'Full state info',
        'W/o velocity info of foot',
        'W/o velocity info of foot & leg',
        'W/o velocity info of foot & leg & thigh'
    ]
    model2missingobs = {
        f'idm_{k}': all_labels[k] for k in range(len(all_idm))
    }
    model2missingobs.update({
        f'fdm_{k}': all_labels[k] for k in range(len(all_fdm))
    })

    sns.set_style('white')
    #sns.set_theme('paper')
    #sns.set_palette('Set2')
    fig, ax = plt.subplots(1 , 1, figsize=(5.5, 5), tight_layout=True)
    df_performance, df_idm, df_fdm = [], [], []
    data_idm = []
    for k in range(len(all_idm)):
        
        policy_performance = rollout(all_policy[k], all_env[k], 30)
        a_pred_vars = [dist.variance.detach().tolist() for dist in idm_action_seq_dist[f'idm_{k}']]
        obs_pred_vars = [dist.variance.detach().tolist() for dist in fdm_obs_seq_dist[f'fdm_{k}']]
        
        df_performance.append(pd.DataFrame({
            'Return': policy_performance,
            'type': ['Return'] * len(policy_performance),
            'model': [model2missingobs[f'idm_{k}']] * len(policy_performance)
        }))
        df_idm.append(pd.DataFrame({
            'prediction variance': [np.mean(a_pred_var) for a_pred_var in a_pred_vars],
            'pred_type': ["Var(A|S,S')"] * len(a_pred_vars),
            'model': [model2missingobs[f'idm_{k}']] * len(a_pred_vars)
        }))
        df_fdm.append(pd.DataFrame({
            'prediction variance': [np.mean(obs_pred_var) for obs_pred_var in obs_pred_vars],
            'pred_type': ["Var(S'|S,A)"] * len(obs_pred_vars),
            'model': [model2missingobs[f'fdm_{k}']] * len(obs_pred_vars)
        }))

        data_idm.append(np.array([np.mean(a_pred_var) for a_pred_var in a_pred_vars], dtype=np.float64))

    # plot for policy performance
    '''
    ax = axs[0]
    df = pd.concat(df_performance)
    sns.barplot(
        data=df,
        x='type',
        y='Return',
        hue='model',
        ax=ax,
        ci='sd',
        linewidth=2,
        #capsize=.2
    )
    ax.set_xticklabels([''])
    ax.set_title("Policy performance in different setting", fontsize=11)
    ax.set_xlabel('Policy Performance', fontsize=11)
    ax.set_ylabel('Return', fontsize=10)
    '''

    # plot for inverse model prediction
    df = pd.concat(df_idm)
    #ax.boxplot(data_idm)



    sns.boxplot(
        data=df,
        x='pred_type',
        y='prediction variance',
        hue='model',
        ax=ax,
        #ci='sd',
        #width=0.2,
        linewidth=2
    )

    ax.set_xticklabels([''])
    ax.legend().set_title('')
    ax.set_title("Prediction variance of IDMs in different state settings", fontsize=12)
    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel('Prediction Variance of IDM', fontsize=12)

    adjust_box_widths(fig, 0.3)

    # plot for inverse model prediction
    '''
    ax = axs[2]
    df = pd.concat(df_fdm)
    sns.violinplot(
        data=df,
        x='pred_type',
        y='prediction variance',
        hue='model',
        ax=ax,
        #ci='sd',
        linewidth=2
    )
    ax.set_xticklabels([''])
    ax.set_title("Variance of S'|S,A in different setting", fontsize=11)
    ax.set_xlabel("Var(S'|S,A)", fontsize=11)
    ax.set_ylabel('Prediction Variance of Forward Dynamics Models', fontsize=10)
    '''
    '''
    for i in range(len(axs)):
        if i == 0:
            axs[i].legend().set_title('')
        else:
            axs[i].legend().remove()
    '''
    plt.show()


if __name__ == '__main__':
    performance_and_model_variance_plot(
        all_path= [
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_[]-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_['foot']-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_['foot', 'leg']-10/",
            "/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/both_model-Walker-missing_['foot', 'leg', 'thigh']-10/",
        ],
        remark= '800000',
    )