from typing import List, Tuple, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
import ray
import pickle



RESULTS_PATH = '/home/xukang/GitRepo/ModelBasedRL/results/missing_obs_test/'
FILE_NAME = [
    'Walker-missing_obs_[]-10-Return.csv',
    "Walker-missing_obs_['foot']-10-Return.csv",
    "Walker-missing_obs_['foot', 'leg']-10-Return.csv",
    "Walker-missing_obs_['foot', 'leg', 'thigh']-10-Return.csv"
]
LEGEND_NAME = [
    'missing none obs info',
    'missing foot vel info',
    'missing foot&leg vel info',
    'missing foot&leg&thigh vel info'
]

ALL_TIMESTEPS = list(range(0, 860000, 20000))


def load_csv(file_path: str) -> Dict:
    with open(file_path, "rb") as f:
        x = pd.read_csv(f)
    steps = x['Step'].tolist()
    returns = x['Value'].tolist()
    
    new_steps, new_returns = [], []
    for step in ALL_TIMESTEPS:
        if step in steps:
            idx = steps.index(step)
            
            new_steps.append(step)
            new_returns.append(returns[idx])

    return new_steps, new_returns


def plot_from_file() -> None:
    df_seq = []
    for i, file_name in enumerate(FILE_NAME):
        steps, returns = load_csv(RESULTS_PATH + file_name)

        df_seq.append(
            pd.DataFrame({
                'Timesteps': steps,
                'Return': returns,
                'Missing Info': [LEGEND_NAME[i]] * len(steps)
            })
        )
    df = pd.concat(df_seq)

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(5, 5))

    sns.lineplot(x='Timesteps', y='Return', data=df, hue='Missing Info', ax=ax)
    ax.set_title(f'Performance of Policies trained in difference observation setting', fontsize=11)
    ax.set_xlabel('Total Steps',fontdict={'size': 11})
    ax.set_ylabel('Return',fontdict={'size': 11})
    ax.legend().set_title('')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    plt.show()    


if __name__ == '__main__':
    plot_from_file()