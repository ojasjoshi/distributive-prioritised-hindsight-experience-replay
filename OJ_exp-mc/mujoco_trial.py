import numpy as np
import gym
import random


# env = gym.make('FetchSlide-v0')
# env = gym.make('FetchReach-v0')
# env = gym.make('FetchPush-v0')
env = gym.make('FetchPickAndPlace-v0')
# env = gym.make('HalfCheetah-v2')
# print(env.reset()['observation'].shape)


env = gym.wrappers.FlattenDictWrapper(
    env, dict_keys=['observation','desired_goal'])
# print(env.observation_space.shape)
obs = env.reset()
print(env.name())
# print(obs.shape,env.action_space)
# env = gym.wrappers.FlattenDictWrapper(
    # env, dict_keys=['observation'])
done = False

def policy(observation, desired_goal=None):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    # print(env.observation_space.shape[0],env.action_space.shape[0])
    return env.action_space.sample()

while not done:
	# env.render()
	# action = policy(obs['observation'], obs['desired_goal'])
	# print("Current Observation is:" ,obs['observation'],obs['desired_goal'])
	# print("Current Observation is: ",obs[-6:-3])
	# print("Location:", obs['observation'][3:6], "Achieved Goal: ",obs['achieved_goal'])
	action = policy(obs)
	# print(action)
	obs, reward, done, info = env.step(action)
	
	# print(obs['observation'][3:6] - obs['achieved_goal'])
	# print(obs[-3:])

	# If we want, we can substitute a goal here and re-compute
	# the reward. For instance, we can just pretend that the desired
	# goal was what we achieved all along.
	# substitute_goal = obs[3:6].copy()
	# print(substitute_goal)
	# substitute_reward = env.compute_reward(
	    # obs[3:6], substitute_goal, info)
	# print('reward is {}, substitute_reward is {}'.format(
	    # reward, substitute_reward))	    