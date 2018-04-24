import numpy as np
import sys
import argparse
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

def parse_arguments():
	parser = argparse.ArgumentParser()
	""" WARNING: With env, change ddpg.py/add_HER function accordingly """
	parser.add_argument('--env', dest='ENV_NAME',type=str, default='FetchPush-v0',help="Environment Name")
	parser.add_argument('--her', dest='HER',type=bool, default=True,help="Do Hindsight Experience Replay")
	parser.add_argument('--per', dest='PER',type=bool, default=True,help="Do Prioritised Experience Replay")
	parser.add_argument('--k', dest='K',type=int, default=4,help="HER parameter")
	parser.add_argument('--her_strategy', dest='her_strategy',type=str, default='future',help="HER strategy")
	parser.add_argument('--batch_size', dest='batch_size',type=int, default=64,help="Batch Size")
	parser.add_argument('--gamma', dest='gamma',type=float, default=0.99,help="Value of gamma")
	parser.add_argument('--soft_target_update', dest='soft_update',type=float, default=1e-3,help="Value of soft update of target network")
	parser.add_argument('--actor_lr', dest='actor_lr',type=float, default=5e-4,help="Value of actor learning rate")
	parser.add_argument('--critic_lr', dest='critic_lr',type=float, default=1e-3,help="Value of critic learning rate")
	parser.add_argument('--alpha', dest='alpha',type=float, default=0.7,help="Value of alpha PER between 0 and 1")
	parser.add_argument('--beta', dest='beta',type=float, default=0.5,help="Value of beta PER between 0 and 1")
	parser.add_argument('--memory_size', dest='memory_size',type=int, default=50000,help="Experience Replay Size")
	parser.add_argument('--max_step_episode', dest='max_step_episode',type=int, default=50,help="Number of steps before resetting the episode")
	parser.add_argument('--file_interval', dest='file_interval',type=int, default=10000,help="Data save interval")
	parser.add_argument('--nb_train_steps', dest='nb_train_steps',type=int, default=200000,help="Number of training steps")
	parser.add_argument('--nb_test_episodes', dest='nb_test_episodes',type=int, default=5,help="Number of test episodes")
	parser.add_argument('--monitor', dest='monitor',type=int, default=False, help="Turn on monitor")
	parser.add_argument('--seed', dest='seed',type=int, default=123, help="Seed")
	parser.add_argument('--delta_clip', dest='delta_clip',type=float, default=np.inf, help="Huber loss delta clip")

	return parser.parse_args()

""" TODO: delta_clip?, rank-based PER, check MujocoProcessor """

args = parse_arguments()

gym.undo_logger_setup()

# Get the environment and extract the number of actions.
env = gym.make(args.ENV_NAME)

# FetchSlide: ([:25]=state ([3:6]=achieved_goal), [-3:]=desired_goal)
# FetchReach: ([:13]=state ([0:3]=achieved_goal), [-3:]=desired_goal)
env = gym.wrappers.FlattenDictWrapper(
    env, dict_keys=['observation', 'desired_goal']) 

if(args.monitor):
	env = wrappers.Monitor(env, '/tmp/{}'.format(ENV_NAME), force=True)

np.random.seed(args.seed)
env.seed(args.seed)

assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Actor model
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))		# (observation_space.shape = (25,) )
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
# print(actor.summary())

# Crtic Model
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

if(args.HER==True and args.PER==False):
	memory = NonSequentialMemory(limit=args.memory_size, window_length=1)
elif(args.HER==True and args.PER==True):
	memory = PrioritisedNonSequentialMemory(limit=args.memory_size, alpha=args.alpha, beta=args.beta, window_length=1) ## 'proportional' priority replay implementation
else:
	print("\nRun vanilla ddpg_mujoco.py for no PER or HER!")
	sys.exit(1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)

## WARNING: make sure memory_interval is 1 for HER to work 
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, batch_size=args.batch_size,
                  delta_clip=args.delta_clip, random_process=random_process, gamma=args.gamma,
                  target_model_update=args.soft_update, do_HER=args.HER, K=args.K, HER_strategy=args.her_strategy,
                  do_PER=args.PER, epsilon=1e-4, processor=MujocoProcessor())
agent.compile([Adam(lr=args.actor_lr), Adam(lr=args.critic_lr)], metrics=['mae'])

if(args.HER==True and args.PER==False):
	print("\nTraining with Prioritised Hindsight Experience Replay\n")
	save_data_path_local = 'HER/'+args.ENV_NAME+'.json'
elif(args.HER==True and args.PER==True):
	print("\nTraining with Hindsight Experience Replay\n")
	save_data_path_local = 'PHER/'+args.ENV_NAME+'.json'
		
""" Start Training (You can always safely abort the training prematurely using Ctrl + C, *once* ) """
agent.fit(env, nb_steps=args.nb_train_steps, visualize=False, verbose=1, save_data_path=save_data_path_local, file_interval=args.file_interval, nb_max_episode_steps=args.max_step_episode)	

# After training is done, we save the final weights and plot the training graph.
try:
	if(args.HER==True and args.PER==False):
		agent.save_weights('HER/ddpg_{}_weights.h5f'.format(args.ENV_NAME), overwrite=True)
		plot_af(file_path='HER/'+ENV_NAME+'.json',save_file_name='HER/'+args.ENV_NAME+'.png')
	elif(args.HER==True and args.PER==True):
		agent.save_weights('PHER/ddpg_{}_weights.h5f'.format(args.ENV_NAME), overwrite=True)
		plot_af(file_path='PHER/'+args.ENV_NAME+'.json',save_file_name='PHER/'+args.ENV_NAME+'.png')
except KeyboardInterrupt:
	pass

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=args.nb_test_episodes, visualize=True, nb_max_episode_steps=args.max_step_episode)	


