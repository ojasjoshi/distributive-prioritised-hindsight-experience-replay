import numpy as np

import gym
from gym import wrappers

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

import sys
# sys.path.append('/Users/ojasjoshi/Desktop/distributive-prioritised-hindsight-experience-replay/tests/keras-rl/')

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import NonSequentialMemory, PrioritisedNonSequentialMemory	
from rl.random import OrnsteinUhlenbeckProcess
from check_json import plot_af

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)


""" TODO: delta_clip?, rank-based PER, check MujocoProcessor """


""" WARNING: With env, change ddpg.py/add_HER function accordingly """
# ENV_NAME = 'FetchSlide-v0'
# ENV_NAME = 'FetchPush-v0'
# ENV_NAME = 'FetchPickAndPlace-v0'
ENV_NAME = 'FetchReach-v0'
HER = True
PER = False
K = 4	# following future strategy by default (can set 'episode' strategy)

gym.undo_logger_setup()

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
# observation_space_n = env.reset()['observation'].shape

# FetchSlide: ([:25]=state ([3:6]=achieved_goal), [-3:]=desired_goal)
# FetchReach: ([:13]=state ([0:3]=achieved_goal), [-3:]=desired_goal)
env = gym.wrappers.FlattenDictWrapper(
    env, dict_keys=['observation', 'desired_goal']) 
# env = wrappers.Monitor(env, '/tmp/{}'.format(ENV_NAME), force=True)
# print(env.observation_space.shape)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))		# (observation_space.shape = (25,) )
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
# print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

## Important Hyperparameters: alpha, beta, limit, batch_size, delta_clip, K

if(HER==True and PER==False):
	memory = NonSequentialMemory(limit=50000, window_length=1)
elif(HER==True and PER==True):
	memory = PrioritisedNonSequentialMemory(limit=50000, alpha=0.7, beta=0.5, window_length=1) ## 'proportional' priority replay implementation
else:
	print("\nRun vanilla ddpg_mujoco.py for no PER or HER!")
	sys.exit(1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)

## WARNING: make sure memory_interval is 1 for HER to work 
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, batch_size=32, delta_clip=np.inf,
                  random_process=random_process, gamma=.99, target_model_update=1e-3, do_HER=HER, K=K, HER_strategy='future',
                  do_PER=PER, epsilon=1e-4, processor=MujocoProcessor())
agent.compile([Adam(lr=5e-4), Adam(lr=1e-3)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# file_interval: episode interval before data dump
if(HER==True and PER==False):
	save_data_path_local = 'HER/'+ENV_NAME+'.json'
elif(HER==True and PER==True):
	save_data_path_local = 'PHER/'+ENV_NAME+'.json'
agent.fit(env, nb_steps=1500000, visualize=False, verbose=1, save_data_path=save_data_path_local, file_interval=10000)	

# After training is done, we save the final weights and plot the training graph.
if(HER==True and PER==False):
	agent.save_weights('HER/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
	plot_af(file_path='HER/'+ENV_NAME+'.json',save_file_name='HER/'+ENV_NAME+'.png')
elif(HER==True and PER==True):
	agent.save_weights('PHER/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
	plot_af(file_path='PHER/'+ENV_NAME+'.json',save_file_name='PHER/'+ENV_NAME+'.png')

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
