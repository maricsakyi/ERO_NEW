# -*- coding: utf-8 -*-
"""
Created on Sat May 29 21:53:49 2021

@author: maricsakyi
"""
import gym

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import PPO2

env = gym.make('CartPole-v1') 
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)


for i in range(10):
    obs = env.reset()
    score=0
    for k in range(200):    
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        
        score+=rewards
        env.render()
        
        if dones:
            break
        
    print("episode:{} reward:{}".format(i,score))
    
import numpy as np

k=np.random.randint(1,50,10)
priority=k/np.sum(k)
print(k)
print(priority)
values=zip(k,priority)
print(values)

for x, y in values:
    print (x,y)