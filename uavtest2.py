from stable_baselines3 import PPO
import os
from UavEnv3dof2 import UavEnv
import time
from stable_baselines3.common.env_checker import check_env
import numpy as np
from funcs import*

models_dir = "models/1673130087"
model_path = f"{models_dir}/3300000.zip"
env = UavEnv()
# If the environment don't follow the interface, an error will be thrown

obs = env.reset(0, 0, train=False)
model = PPO.load(model_path,env=env)

episodes = 100
for episode in range(episodes):
    obs = env.reset(0, 0, train=False)
    done = False
    print("End of episode")
    while not done:
        action,_states = model.predict(obs)
        obs, rewards, done, info = env.step(action, test = True)
        
        #print(obs)
        """print("State", obs)
        print("Action", action)
        print(rewards)"""
