import os
import sys
import numpy as np
import gym
import argparse
from DDPG_baseline_v2.baselines.ddpg.main_config import run_ddpg
from DDPG_baseline_v2.baselines.ddpg.configs.config import ddpg_config
from utils import *


sys.path.append('./')


saving_folder = './results/'
trial_id = 1
nb_runs = 1
#env_id = 'HalfCheetah-v2'
#env_id = 'InvertedPendulum-v2'
#env_id = 'InvertedDoublePendulum-v2'
#env_id = 'Hopper-v2'
#env_id='Hopper-v3'
#env_id = 'humanoid-v3'

#env_id ='MountainCarContinuous-v0'
env_id = 'Pendulum-v0'
#env_id='CarRacing-v0'
#env_id='CartPole-v0'
#env_id='LunarLander-v2'
#env_id='LunarLanderContinuous-v2'
#env_id='BipedalWalker-v3'

ddpg_noise = 'ou_0.2'# 'adaptive_param_0.2' #
rollout = 1000


#ddpg_noise = 'adaptive-param_0.2'
seed = 0
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def run_experiment(env_id, trial, noise_type, saving_folder, rollout, seed):

    print("seed:",seed)
    np.random.seed(seed)
    data_path = saving_folder

    # create data path
    #data_path = create_data_path(saving_folder, env_id, trial_id)
    # load ddpg config
    dict_args = ddpg_config(env_id=env_id,
                            data_path=data_path,
                            noise=noise_type,
                            trial_id=trial_id,
                            seed=int(np.random.random() * 1e6),
                            buffer_location=None,
                            rollout_steps = rollout 
                            )

    run_ddpg(dict_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial', type=int, default=trial_id)
    parser.add_argument('--env_id', type=str, default=env_id)
    parser.add_argument('--rollout', type=int, default=rollout)
    parser.add_argument('--noise_type', type=str, default=ddpg_noise)  # choices are adaptive-param_xx, ou_xx, normal_xx, decreasing-ou_xx, none
    parser.add_argument('--saving_folder', type=str, default=saving_folder)
    parser.add_argument('--seed', type=int, default=seed)
    args = vars(parser.parse_args())

    perf = np.zeros([nb_runs])
    for i in range(nb_runs):

        perf[i] = run_experiment(**args)
        print(perf)
        print('Average performance: ', perf.mean())
