#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Lambda, Add, Subtract
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from gym.wrappers import Monitor
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import collections
import time
import math
import shutil
import pickle
import random

class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, env, model=None):

		self.learning_rate = 0.0001																							#HYPERPARAMETER1

		if(model==None):
			## dueling network
			print("Setting up Dueling DDQN network....")
			inp = Input(shape=(env.observation_space.shape[0]+1,))
			layer_shared = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
			layer_shared = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			layer_shared = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			print("Shared layers initialized....")

			layer_v = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			layer_a = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			layer_v = Dense(1,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_v)
			layer_a = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_a)
			print("Value and Advantage Layers initialised....")

			layer_mean = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(layer_a)
			temp = layer_v
			temp2 = layer_mean

			for i in range(env.action_space.n-1):
				layer_v = keras.layers.concatenate([layer_v,temp],axis=-1)
				layer_mean = keras.layers.concatenate([layer_mean,temp2],axis=-1)

			layer_q = Subtract()([layer_a,layer_mean])
			layer_q = Add()([layer_q,layer_v])

			self.model = Model(inp, layer_q)
			print("Q-function layer initialized.... :)\n")
			
			#deep network
			# inp = Input(shape=(env.observation_space.shape[0]+1,))
			# hidden_layer = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
			# hidden_layer = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# hidden_layer = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# hidden_layer = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# output_layer = Dense(env.action_space.n,activation='linear', kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# self.model = Model(inp,output_layer)
			# print("Q-Network initialized.... :)\n")			
		else:
			self.model = model
			
		# self.model.summary()
		self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
		# plot_model(self.model, to_file='graphs/Duel_DQN.png', show_shapes = True)			

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		self.model.save_weights(suffix)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.model = keras.models.load_model(model_file)
	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		self.model.set_weights(weight_file)

	def visualise_weights(self):
		print("Current Weights\n")
		for layer in self.model.layers:
			temp = layer.get_weights()
			print(temp)


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.burn_in = burn_in
		self.memory_size = memory_size
		self.experience = collections.deque()
		self.batch = []
		self.priority = []

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		# indices = np.random.randint(0,len(self.experience),batch_size)
		# if(len(priority)<batch_size):
			# self.priority = list(np.arange(batch_size))
		probabilities = np.asarray(self.priority)/np.sum(self.priority)
		indices = np.random.choice(len(self.experience),batch_size,p=probabilities,replace=False)
		self.batch = [self.experience[i] for i in indices]

	def append(self, transition, weight):
		# Appends transition to the memory.
		if(len(self.experience)>self.memory_size):
			pop = self.experience.popleft()
			self.priority = self.priority[1:]
		self.experience.append(transition)
		self.priority.append(weight)


class DQN_Agent():

	def __init__(self, env, K=0, pickle_path='.', model=None, render=False):

		self.net = QNetwork(env,model)
		self.prediction_net = QNetwork(env,model)
		self.env = env
		self.replay_mem = Replay_Memory()																	    #HYPERPARAMETER2
		self.render = render
		self.feature_size = env.observation_space.shape[0]+1
		self.action_size = env.action_space.n
		self.discount_factor = 1 	

		self.epsilon = 0.75																						#HYPERPARAMETER3
		self.epsilon_min = 0.05																					#HYPERPARAMETER4
		self.num_episodes = 4000
		self.epsilon_decay = float((self.epsilon-self.epsilon_min)/150000)										#HYPERPARAMETER5
		self.avg_rew_buf_size_epi = 10 
		self.save_model_episodes = 100
		self.print_epi = 1 
		self.p = [0.4,0.6]
		self.p_decay = 6e-4
		self.main_goal = np.random.choice([0.5,-1.2],1,p=self.p)
		self.K = K
		self.pickle_path = pickle_path
		self.opt_steps = 100
		self.experience_priority = 10	#priority to choose actual experience over hindsight experience 		#HYPERPARAMETER6
		#stores the test reward
		self.evaluate = 0.0

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		if(np.random.random_sample()<self.epsilon):
			return np.random.randint(self.action_size)
		else:
			return np.argmax(q_values[0])

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		return np.argmax(q_values[0])
		

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.
		folder = None
		test_reward = []		#saves the testing rewards
		train_reward = []		#saves the training rewards
		curr_episode = 1 		
		iters = 1
		max_reward = 0
		reward_buf = collections.deque()
		
		self.burn_in_memory()
		print("Burnin done....")	
		print("HER with K = ",self.K)
		save_episode_id=np.around(np.linspace(0,self.num_episodes,num=60))

		# #saving video file
		# video_file_path = 'online_k'+str(self.K)+'/videos/'
		# self.env = Monitor(self.env,video_file_path,video_callable= lambda episode_id: episode_id in save_episode_id, force=True)
			
		# learning iterations

		for e in range(self.num_episodes):																			#uncomment for mountaincar
			episode_experience = []
			curr_reward = 0
			curr_state = self.env.reset()	
			current_goal = self.main_goal[0]																		#sample current goal
			temp_state = np.append(curr_state,current_goal)
			temp_state = temp_state.reshape([1,temp_state.shape[0]])
			curr_action = self.epsilon_greedy_policy(self.net.model.predict(temp_state))
			# curr_action = np.random.randint(self.action_size)

			while(True): 																							#uncomment for mountaincar
				if(self.render==True):
					self.env.render()

				nextstate, reward, is_terminal, _ = self.env.step(curr_action)
				curr_reward += reward
				# self.replay_mem.append([curr_state,curr_action,reward,nextstate,is_terminal])
				episode_experience.append([curr_state,curr_action,reward,nextstate,self.main_goal,is_terminal])
				if(is_terminal):
					break
				
				#Adding original experience
				temp_state = np.append(curr_state,current_goal)
				temp_nextstate = np.append(nextstate,current_goal)
				self.replay_mem.append([temp_state.reshape(1,temp_state.shape[0]),curr_action,reward,temp_nextstate.reshape(1,temp_nextstate.shape[0]),is_terminal],self.experience_priority)			

				#optimisation step
				self.replay_mem.sample_batch()
				state_t = np.squeeze(np.asarray([x[0] for x in self.replay_mem.batch]))
				action_t = np.squeeze(np.asarray([x[1] for x in self.replay_mem.batch]))
				reward_t = np.squeeze(np.asarray([x[2] for x in self.replay_mem.batch]))
				nextstate_t = np.asarray([x[3] for x in self.replay_mem.batch])

				input_state = state_t
				truth = self.net.model.predict(state_t)

				for i in range(len(self.replay_mem.batch)):

					if(self.replay_mem.batch[i][4]==True):
						truth[i][action_t[i]] = reward_t[i]
					else:
						# q_target = reward_t[i] + self.discount_factor*np.amax(self.prediction_net.model.predict(nextstate_t[i])) #DQN
						q_target = reward_t[i] + self.discount_factor*self.prediction_net.model.predict(nextstate_t[i])[0,np.argmax(truth[i])] #DDQN
						truth[i][action_t[i]] = q_target
				
				history = self.net.model.fit(input_state,truth,epochs=1,verbose=0,batch_size = len(self.replay_mem.batch))
				loss = history.history['loss']

				#going to next step of episode
				temp = np.append(nextstate,current_goal)
				temp = temp.reshape(1,temp.shape[0])
				q_nextstate = self.net.model.predict(temp)
				nextaction = self.epsilon_greedy_policy(q_nextstate)
				curr_state = nextstate
				curr_action = nextaction

				self.epsilon -= self.epsilon_decay
				self.epsilon = max(self.epsilon, 0.05)

			## Adding HER 
			for t in range(len(episode_experience)):
					s,a,r,ns,_,it = episode_experience[t]
					if(self.K!=0):
						for k in range(self.K): #loop over size of augmented transitions
							new_g = np.random.randint(t,len(episode_experience)) 	#future_strategy from HER 
							_,_,_,ng,_,_ = episode_experience[new_g]
							r_n = 0 if np.sum(ns==ng)==self.feature_size+1 else -1
							it_n = True if np.sum(ns==ng)==self.feature_size+1 else False
							temp_state = np.append(s,ng[0])
							temp_nextstate = np.append(ns,ng[0])
							self.replay_mem.append([temp_state.reshape(1,temp_state.shape[0]),a,r_n,temp_nextstate.reshape(1,temp_nextstate.shape[0]),it_n],1)

			## changing goal sampling bias
			self.p[0] = min(1,self.p[0]+self.p_decay)
			self.p[1] = max(0,self.p[1]-self.p_decay)

			# shitty things
			if(e%self.save_model_episodes==0):
				with open(self.pickle_path+'train_rew_backup.pkl', 'wb') as f:
					pickle.dump(train_reward, f)
					
				self.prediction_net.model.save(self.pickle_path+'prediction_model_latest.h5')
				test_mean, test_std = self.test(self.env)
				print("Test Mean = {}, Test Std = {}".format(test_mean, test_std))
				self.render_one_episode(self.env)
			###end of episode##

			self.prediction_net.load_model_weights(self.net.model.get_weights())

			## Evaluating training session
			max_reward = max(max_reward, curr_reward)

			if(len(reward_buf)>self.avg_rew_buf_size_epi):
				reward_buf.popleft()
			reward_buf.append(curr_reward)
			avg_reward = sum(reward_buf)/len(reward_buf)

			if(curr_episode%self.print_epi==0):
				print(curr_episode, self.epsilon ,int(avg_reward), int(curr_reward), loss)
			curr_episode += 1

			train_reward.append(curr_reward)

		return train_reward

	def render_one_episode(self,env):
		state = env.reset()
		current_goal = self.main_goal[0]
		state = np.append(state,current_goal)
		state = state.reshape([1,env.observation_space.shape[0]+1])
		action = self.greedy_policy(self.net.model.predict(state))
		while(True):
		    env.render()
		    # print(state)
		    nextstate, reward, is_terminal, _ = env.step(action)
		    if(is_terminal == True):
		        break                  
		    nextstate = np.append(nextstate,current_goal)          
		    state = nextstate.reshape([1,env.observation_space.shape[0]+1])
		    action = self.greedy_policy(self.net.model.predict(state))

	def test(self, env, epi=10):

		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.

		tot_reward = []
		for e in range(epi):	
			curr_reward = 0															
			state = env.reset()
			current_goal = self.main_goal[0]
			state = np.append(state,current_goal)
			state = state.reshape([1,env.observation_space.shape[0]+1])
			action = self.greedy_policy(self.net.model.predict(state))
			while(True):
			    nextstate, reward, is_terminal, _ = env.step(action)
			    curr_reward += reward
			    if(is_terminal == True):
			        break                  
			    nextstate = np.append(nextstate,current_goal)          
			    state = nextstate.reshape([1,env.observation_space.shape[0]+1])
			    action = self.greedy_policy(self.net.model.predict(state))
			tot_reward.append(curr_reward)

		self.evaluate = float(np.sum(tot_reward)/epi)

		return np.mean(tot_reward), np.std(tot_reward)

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		# Burn-in with random state and action transitions
		curr_mem_size = 0

		episode_experience = []
		state = self.env.reset()
		action = np.random.randint(self.action_size)
		while(curr_mem_size<self.replay_mem.burn_in):
			nextstate, reward, is_terminal, _ = self.env.step(action)
			episode_experience.append([state,action,reward,nextstate,self.main_goal,is_terminal])
			if(is_terminal == True):
				for t in range(len(episode_experience)):
					s,a,r,ns,g,it = episode_experience[t]
					temp_state = np.append(s,g[0])
					temp_nextstate = np.append(ns,g[0])
					self.replay_mem.append([temp_state.reshape(1,temp_state.shape[0]),a,r,temp_nextstate.reshape(1,temp_nextstate.shape[0]),it],self.experience_priority)

					#HER
					if(self.K!=0):
						for k in range(self.K): #loop over size of augmented transitions
							new_g = np.random.randint(t,len(episode_experience)) 	#future_strategy from HER Paper
							_,_,_,ng,_,_ = episode_experience[new_g]
							r_n = 0 if np.sum(ns==ng)==self.feature_size*2 else -1
							it_n = True if np.sum(ns==ng)==self.feature_size*2 else False
							temp_state = np.append(s,ng[0])
							temp_nextstate = np.append(ns,ng[0])
							self.replay_mem.append([temp_state.reshape(1,temp_state.shape[0]),a,r_n,temp_nextstate.reshape(1,temp_nextstate.shape[0]),it_n],1)
					curr_mem_size += (1+self.K)									
				state = self.env.reset()
			else:
				state = nextstate
			action = np.random.randint(self.action_size)


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str,default='MountainCar-v0')
	parser.add_argument('--render',dest='render',type=bool,default=False)
	parser.add_argument('--train',dest='train',type=bool,default=True)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--deep',dest='deep',type=bool,default=False)
	parser.add_argument('--duel',dest='duel',type=bool,default=False)
	parser.add_argument('--replay',dest='replay',type=bool,default=False)
	parser.add_argument('--K',dest='K',type=int,default=0)
	parser.add_argument('--pickle_dir',dest='pickle_dir',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	env = gym.make(environment_name)

	#Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	model = None
	# model = keras.models.load_model('prediction_model_latest.h5')        # if loading from my_saved_weights
	# print(model)
	agent = DQN_Agent(env,args.K,args.pickle_dir,model,args.render)

	if(args.train==True):
		train_reward,final_weight_file = agent.train()
	
	# if(args.train==False):
		# agent.test(args.model)
		# print(agent.evaluate)


if __name__ == '__main__':
	main(sys.argv)
