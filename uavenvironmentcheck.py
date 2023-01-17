from UavEnv3dof2 import UavEnv
import time

env = UavEnv()
episodes = 100

for episode in range(episodes):
	done = False
	obs = env.reset()
	#time.sleep(3)
	while not done:#not done:
		random_action = env.action_space.sample()
		#print("action",random_action)
		obs, reward, done, info = env.step(random_action, test = True)
		#print('reward',reward)
		"""print(env.Aircraft.aic)
		print(reward)"""
