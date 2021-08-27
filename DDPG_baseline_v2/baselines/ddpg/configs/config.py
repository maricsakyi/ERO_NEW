import numpy as np

def ddpg_config(env_id,
                data_path,
                noise='ou_0.3',
                trial_id=999,
                seed=int(np.random.random()*1e6),
                nb_epochs=1000,
                buffer_location=None,
                rollout_steps=200
                ):

    args_dict = dict(env_id=env_id,
                     render_eval=False,
                     layer_norm=False,
                     render=False,
                     normalize_returns=False,
                     normalize_observations=True,
                     seed=seed, #int(np.random.random()*1e6),
                     critic_l2_reg=0,
                     batch_size=64,  # per MPI worker
                     actor_lr=1e-4,
                     critic_lr=1e-3,
                     popart=False,
                     gamma=0.99,
                     reward_scale=1.,
                     clip_norm=None,
                     nb_timesteps=2e6, # with default settings, perform 2M steps total
                     nb_epoch_cycles=20,
                     nb_train_steps=50, # per epoch cycle and MPI worker
                     nb_eval_steps=1000, # per epoch cycle and MPI worker
                     nb_eval_episodes=10,
                     nb_rollout_steps=rollout_steps, # per epoch cycle and MPI worker
                     noise_type=noise,  # choices are adaptive-param_xx, ou_xx, normal_xx, decreasing-ou_xx, none
                     evaluation=True,
                     buffer_location=buffer_location,
                     data_path=data_path,
                     trial_id=trial_id,
                     max_memory=1e6,
                   
                     meta_lr=1e-4,
                     meta_batch_size=64,
                     nb_meta_steps=1,
                     )
    if env_id=='MountainCarContinuous-v0':
        args_dict['nb_epochs'] = 250
             
    return args_dict
