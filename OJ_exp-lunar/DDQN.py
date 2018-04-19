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

## Possible additions/experiments:

#BatchNorm
#Dueling architechture
#Scaling rewards or not?
#LR
#soft target updates
#memory size and batch size(bigger the better?)
#epsilon (intuition is of smaller values)
#self.train_interval

##HER 
# prioritising original replay
# sampling bias for main goal

def huber_loss_custom(y_true,y_pred):
	return tf.losses.huber_loss(y_true,y_pred)

class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, env, duel=True, dueling_type='avg', model=None):

		self.learning_rate = 1e-4																						#HYPERPARAMETER1
		self.enable_dueling_network = duel
		self.dueling_type = dueling_type

		if(model==None):
			## dueling network
			# print("Setting up Dueling DDQN network....")
			# inp = Input(shape=(env.observation_space.shape[0],))
			# layer_shared = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
			# layer_shared = BatchNormalization()(layer_shared)
			# layer_shared = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			# layer_shared = BatchNormalization()(layer_shared)
			# layer_shared = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			# layer_shared = BatchNormalization()(layer_shared)
			# print("Shared layers initialized....")

			# layer_v = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			# layer_shared = BatchNormalization()(layer_shared)
			# layer_a = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared)
			# layer_shared = BatchNormalization()(layer_shared)
			# layer_v = Dense(1,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_v)
			# layer_a = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_a)
			# print("Value and Advantage Layers initialised....")

			# layer_mean = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(layer_a)
			# temp = layer_v
			# temp2 = layer_mean

			# for i in range(env.action_space.n-1):
			# 	layer_v = keras.layers.concatenate([layer_v,temp],axis=-1)
			# 	layer_mean = keras.layers.concatenate([layer_mean,temp2],axis=-1)

			# layer_q = Subtract()([layer_a,layer_mean])
			# layer_q = Add()([layer_q,layer_v])

			# self.model = Model(inp, layer_q)
			# print("Q-function layer initialized.... :)\n")
			
			## deep network
			inp = Input(shape=(env.observation_space.shape[0],))

			#alternate1
			hidden_layer = Dense(64,activation='relu',kernel_initializer='he_uniform')(inp)										#(CHANGED)
			hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform')(inp)
			hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform')(inp)
			
			#alternate2
			# hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
			# hidden_layer = BatchNormalization()(hidden_layer)
			# hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# hidden_layer = BatchNormalization()(hidden_layer)
			# hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# hidden_layer = BatchNormalization()(hidden_layer)
			# hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			# hidden_layer = BatchNormalization()(hidden_layer)
			# hidden_layer = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(hidden_layer)
			
			output_layer = Dense(env.action_space.n,activation='linear', kernel_initializer='he_uniform')(hidden_layer)
			self.model = Model(inp,output_layer)
			print("Q-Network initialized.... :)\n")			

		else:
			self.model = model

		if self.enable_dueling_network:
			layer = self.model.layers[-2]
			nb_action = self.model.output._keras_shape[-1]	         
			y = Dense(nb_action + 1, activation='linear')(layer.output)
			if self.dueling_type == 'avg':
			    outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(y)
			elif self.dueling_type == 'max':
			    outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(y)
			elif self.dueling_type == 'naive':
			    outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
			else:
			    assert False, "dueling_type must be one of {'avg','max','naive'}"

			self.model = Model(inputs=self.model.input, outputs=outputlayer)

		self.model.compile(optimizer = Adam(lr=self.learning_rate), loss=huber_loss_custom)		

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

	def __init__(self, memory_size=100000, burn_in=2000):													    #(CHANGED)

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.burn_in = burn_in
		self.memory_size = memory_size
		self.experience = collections.deque()
		self.batch = []

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		# indices = np.random.randint(0,len(self.experience),batch_size)
		# if(len(priority)<batch_size):
			# self.priority = list(np.arange(batch_size))	
		indices = np.random.randint(0,len(self.experience),batch_size)
		self.batch = [self.experience[i] for i in indices]

	def append(self, transition):
		# Appends transition to the memory.
		if(len(self.experience)>self.memory_size):
			pop = self.experience.popleft()
		self.experience.append(transition)


class DQN_Agent():

	def __init__(self, env, K=0, pickle_path='.', model=None, render=False, tau=1., clip=(-500., 500.)):

		self.net = QNetwork(env,model)
		self.prediction_net = QNetwork(env,model)
		self.env = env
		self.replay_mem = Replay_Memory()																	    #HYPERPARAMETER2
		self.render = render
		self.feature_size = env.observation_space.shape[0]
		self.action_size = env.action_space.n
		self.discount_factor = 0.99

		self.epsilon = 0.1																					#HYPERPARAMETER3 (CHANGED)
		self.epsilon_min = 0.00																					#HYPERPARAMETER4
		self.num_episodes = 50000
		# self.epsilon_decay = float((self.epsilon-self.epsilon_min)/self.num_episodes)							#HYPERPARAMETER5 (CHANGED)
		self.epsilon_decay = 0
		self.avg_rew_buf_size_epi = 10 
		self.save_model_episodes = 25
		self.render_one_episode_interval = 25
		self.print_epi = np.inf
		self.pickle_path = pickle_path
		self.opt_steps = 100
		self.experience_priority = 10	#priority to choose actual experience over hindsight experience 		#HYPERPARAMETER6
		#stores the test reward
		self.evaluate = 0.0
		self.test_model_episodes = 25
		self.TAU = 1e-2																							#reduce (CHANGED)

		self.tau = tau 	# (boltzman_policy)
		self.clip = clip

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		if(np.random.random_sample()<self.epsilon):
			return np.random.randint(self.action_size)
		else:
			return np.argmax(q_values[0])

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		return np.argmax(q_values[0])

	def boltzman_policy(self, q_values):
		q_values[0] = q_values[0].astype('float64')
		nb_actions = q_values[0].shape[0]

		exp_values = np.exp(np.clip(q_values[0] / self.tau, self.clip[0], self.clip[1]))
		probs = exp_values / np.sum(exp_values)
		action = np.random.choice(range(nb_actions), p=probs)
		return action
		

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
		train_data = []
		q_vals_list = []
		num_landings = 0
		loss_list = []

		self.burn_in_memory()
		print("Burnin done....")	

		# #saving video file
		# save_episode_id=np.around(np.linspace(0,self.num_episodes,num=500))
		# video_file_path = 'ddqn_vanilla_batch500/videos/'
		# self.env = Monitor(self.env,video_file_path,video_callable= lambda episode_id: episode_id in save_episode_id, force=True)
		
		# learning iterations
		for e in range(self.num_episodes):																			#uncomment for mountaincar
			episode_experience = []
			curr_reward = 0
			curr_state = self.env.reset()	
			temp_state = curr_state
			temp_state = temp_state.reshape([1,self.env.observation_space.shape[0]])
			# curr_action = self.epsilon_greedy_policy(self.net.model.predict(temp_state))
			curr_action = self.boltzman_policy(self.net.model.predict(temp_state))
			curr_steps = 0

			while(True): 																							#uncomment for mountaincar
				if(self.render==True):
					self.env.render()

				nextstate, reward, is_terminal, _ = self.env.step(curr_action)
				curr_reward += reward
				# self.replay_mem.append([curr_state,curr_action,reward,nextstate,is_terminal])
				self.replay_mem.append([curr_state.reshape([1,self.env.observation_space.shape[0]]),curr_action,reward,nextstate.reshape([1,self.env.observation_space.shape[0]]),is_terminal])	#CHANGED		
				
				q_vals_list.append(self.net.model.predict(curr_state.reshape([1,self.env.observation_space.shape[0]]))[0][curr_action])

				if(is_terminal):
					last_reward = reward
					if(last_reward==100):
						num_landings +=1
					break		

				#optimisation step
				self.replay_mem.sample_batch(32)
				state_t = np.asarray([np.squeeze(x[0]) for x in self.replay_mem.batch])
				action_t = np.squeeze(np.asarray([x[1] for x in self.replay_mem.batch]))
				reward_t = np.squeeze(np.asarray([x[2] for x in self.replay_mem.batch]))
				nextstate_t = np.asarray([x[3] for x in self.replay_mem.batch])
				# print(state_t.shape,action_t.shape,reward_t.shape,nextstate_t.shape)

				input_state = state_t
				truth = self.net.model.predict(state_t)
				for i in range(len(self.replay_mem.batch)):
					if(self.replay_mem.batch[i][4]==True):
						truth[i][action_t[i]] = reward_t[i]
					else:					
						# q_target = reward_t[i] + self.discount_factor*np.amax(self.prediction_net.model.predict(nextstate_t[i])) #DQN
						q_target = reward_t[i] + self.discount_factor*self.prediction_net.model.predict(nextstate_t[i])[0,np.argmax(truth[i])] #DDQN
						truth[i][action_t[i]] = q_target
				
				history = self.net.model.fit(input_state,truth,epochs=1,verbose=0,batch_size=len(self.replay_mem.batch))
				loss = history.history['loss']
				loss_list.append(loss)

				#going to next step of episode
				nextstate = nextstate.reshape([1,self.env.observation_space.shape[0]])
				q_nextstate = self.net.model.predict(nextstate)
				# nextaction = self.epsilon_greedy_policy(q_nextstate)
				nextaction = self.boltzman_policy(q_nextstate)
				curr_state = nextstate
				curr_action = nextaction

				curr_steps += 1
			###end of episode###

			#decay epsilon
			self.epsilon -= self.epsilon_decay
			self.epsilon = max(self.epsilon, 0.0)

			## (optimisation step can be here)

			# shitty things
			if(e%self.test_model_episodes==0):
				test_mean, test_std, mean_q_test = self.test(self.env)
				train_data.append([test_mean, test_std])
				template = "Episode: {episode}, Test_Mean: {test_mean:.3f}, Test_Std: {test_std:.3f}, Mean_Q_val_train: {mean_q_tr:.3f}, Mean_Q_val_test: {mean_q_te:.3f}, Avg_Landings: {avg_land:.3f}, Mean_loss: {mean_loss:.3f}"
				variables = {
					'episode': e,
					'test_mean':test_mean,
					'test_std':test_std,
					'mean_q_tr':np.mean(np.hstack(q_vals_list)),
					'mean_q_te':mean_q_test,
					'avg_land':num_landings/self.test_model_episodes,
					'mean_loss':np.mean(loss_list)
				}
				print(template.format(**variables))
				q_vals_list = []
				num_landings = 0
			
			# if(e%self.render_one_episode_interval==0):
				# self.render_one_episode(self.env)

			## add save experience replay to this
			if(e%self.save_model_episodes==0):
				with open(self.pickle_path+'train_rew_backup.pkl', 'wb') as f:
					pickle.dump([test_mean, test_std], f)
				with open(self.pickle_path+'train_rew_backup_validation.pkl', 'wb') as f:
					pickle.dump(train_data, f)
					
				self.prediction_net.model.save(self.pickle_path+'prediction_model_latest.h5')

			## soft updates
			model_weights = self.net.model.get_weights()
			model_target_weights = self.prediction_net.model.get_weights()
			for i in range(len(model_weights)):
				model_target_weights[i] = self.TAU * model_weights[i] + (1 - self.TAU)* model_target_weights[i]

			self.prediction_net.model.set_weights(model_target_weights)
			# self.prediction_net.load_model_weights(self.net.model.get_weights())

			## Evaluating training session
			max_reward = max(max_reward, curr_reward)

			if(len(reward_buf)>self.avg_rew_buf_size_epi):
				reward_buf.popleft()
			reward_buf.append(curr_reward)
			avg_reward = sum(reward_buf)/len(reward_buf)

			if(curr_episode%self.print_epi==0):
				# print("Episode: {}, Epsilon: {}, Average Rew: {}, Episode Rew: {}, Loss: {}, Steps: {}, Last Rew: {}".format(curr_episode, self.epsilon ,int(avg_reward), int(curr_reward), loss, curr_steps, last_reward))
				print("Episode: {}, Average Rew: {}, Episode Rew: {}, Loss: {}, Steps: {}, Last Rew: {}".format(curr_episode ,int(avg_reward), int(curr_reward), loss, curr_steps, last_reward))
			curr_episode += 1

			train_reward.append(curr_reward)

		return train_reward

	def render_one_episode(self,env):
		state = env.reset()
		state = state.reshape([1,env.observation_space.shape[0]])
		# action = self.greedy_policy(self.net.model.predict(state))
		action = self.boltzman_policy(self.net.model.predict(state))
		while(True):
		    env.render()
		    # print(state)
		    nextstate, reward, is_terminal, _ = env.step(action)
		    if(is_terminal == True):
		        break                  
		    state = nextstate.reshape([1,env.observation_space.shape[0]])
		    # action = self.greedy_policy(self.net.model.predict(state))
		    action = self.boltzman_policy(self.net.model.predict(state))

	def test(self, env, epi=50):

		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.

		tot_reward = []
		q_list = []
		for e in range(epi):	
			curr_reward = 0															
			state = env.reset()
			state = state.reshape([1,env.observation_space.shape[0]])
			# action = self.greedy_policy(self.net.model.predict(state))
			action = self.boltzman_policy(self.prediction_net.model.predict(state))
			while(True):
				q_list.append(self.prediction_net.model.predict(state)[0][action])
				nextstate, reward, is_terminal, _ = env.step(action)
				curr_reward += reward
				if(is_terminal == True):
				    break                  
				state = nextstate.reshape([1,env.observation_space.shape[0]])
				# action = self.greedy_policy(self.net.model.predict(state))
				action = self.boltzman_policy(self.prediction_net.model.predict(state))
			tot_reward.append(curr_reward)

		self.evaluate = float(np.sum(tot_reward)/epi)

		return np.mean(tot_reward), np.std(tot_reward), np.mean(q_list)

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		# Burn-in with random state and action transitions
		curr_mem_size = 0
		episode_experience = []
		state = self.env.reset()
		state = state.reshape([1,self.env.observation_space.shape[0]])						# 1x8
		# action = self.epsilon_greedy_policy(self.net.model.predict(state))
		action = self.boltzman_policy(self.net.model.predict(state))
		while(curr_mem_size<self.replay_mem.burn_in):
			nextstate, reward, is_terminal, _ = self.env.step(action)
			episode_experience.append([state,action,reward,nextstate.reshape([1,self.env.observation_space.shape[0]]),is_terminal])			#CHANGED
			# print(np.asarray(episode_experience[0][0].shape))
			if(is_terminal == True):
				for t in range(len(episode_experience)):
					s,a,r,ns,it = episode_experience[t]
					self.replay_mem.append([s,a,r,ns,it])
					curr_mem_size += 1									
				state = self.env.reset()
			else:
				state = nextstate.reshape([1,self.env.observation_space.shape[0]])
				# action = self.epsilon_greedy_policy(self.net.model.predict(state))
				action = self.boltzman_policy(self.net.model.predict(state))


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str,default='LunarLander-v2')
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
	np.random.seed(123)
	env.seed(123)

	#Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	model = None
	# model = keras.models.load_model('ddqn_vanilla_batch500/prediction_model_latest.h5')        # if loading from my_saved_weights
	# print(model)
	agent = DQN_Agent(env,args.K,args.pickle_dir,model,args.render)

	if(args.train==True):
		train_reward,final_weight_file = agent.train()
	
	# if(args.train==False):
		# agent.test(args.model)
		# print(agent.evaluate)


if __name__ == '__main__':
	main(sys.argv)
