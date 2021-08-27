import sys
sys.path.append('../../')
import DDPG_baseline_v2.baselines.ddpg.training as training
import argparse
import time
import os
import logging
from DDPG_baseline_v2.baselines import logger, bench
from DDPG_baseline_v2.baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from DDPG_baseline_v2.baselines.ddpg.models import Actor, Critic
#from DDPG_baseline_v2.baselines.ddpg.memory import Memory
from DDPG_baseline_v2.baselines.ddpg.memory import DynamicMemory as Memory
from DDPG_baseline_v2.baselines.ddpg.noise import *
from DDPG_baseline_v2.baselines.common import tf_util as U
import gym
import tensorflow as tf
from mpi4py import MPI
import pickle
import os

def run(env_id, seed, noise_type, layer_norm, evaluation, buffer_location, trial_id, data_path, max_memory,
        nb_eval_episodes, **kwargs):

    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0: logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json" % rank))
    actions_lbound = env.action_space.low
    actions_ubound = env.action_space.high
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json" % rank))
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    nb_observations = env.observation_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'decreasing-ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = DecreasingOrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                                  sigma=float(stddev) * np.ones(nb_actions),
                                                                  decreasing_rate=float(stddev)/float(kwargs['nb_epochs']))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(max_memory),
                    action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape
                    )

    activation_map = { "relu" : tf.nn.relu, "leaky_relu" : U.lrelu, "tanh" :tf.nn.tanh}

    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions,layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.run_agent(env=env, eval_env=eval_env, param_noise=param_noise, action_noise=action_noise,
                       actor=actor, critic=critic, memory=memory, buffer_location = buffer_location,
                       trial_id = trial_id, data_path = data_path, nb_eval_episodes=nb_eval_episodes, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def run_ddpg(dict_args):

    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=dict_args['data_path'])
    #n_timesteps = dict_args['nb_epochs'] * dict_args['nb_epoch_cycles'] * dict_args['nb_rollout_steps']
    # Run actual script.

    run(**dict_args)

