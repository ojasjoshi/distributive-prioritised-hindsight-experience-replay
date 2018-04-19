import numpy as np
import gym
import random


env = gym.make('FetchSlide-v0')
env = gym.wrappers.FlattenDictWrapper(
    env, dict_keys=['observation', 'desired_goal'])
obs = env.reset()
print(obs.shape,env.action_space)
done = False

def policy(observation, desired_goal=None):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    print(env.observation_space.shape[0],env.action_space.shape[0])
    return env.action_space.sample()
    # return np.random.randint(2)

while not done:
	env.render()
	# action = policy(obs['observation'], obs['desired_goal'])
	action = policy(obs)
	print(action)
	obs, reward, done, info = env.step(action)

	# If we want, we can substitute a goal here and re-compute
	# the reward. For instance, we can just pretend that the desired
	# goal was what we achieved all along.
	# substitute_goal = obs['achieved_goal'].copy()
	# substitute_reward = env.compute_reward(
	    # obs['achieved_goal'], substitute_goal, info)
	# print('reward is {}, substitute_reward is {}'.format(
	    # reward, substitute_reward))