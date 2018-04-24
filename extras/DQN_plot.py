import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Lambda, Add, Subtract
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from gym.wrappers import Monitor
import collections
from keras.models import load_model

import numpy as np
import sys, copy, argparse
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pickle
import re
import random

class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, env, replay, deep, duel):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		self.learning_rate = 0.0001																							#HYPERPARAMETER1

		if(deep==False and duel==False):
			print("Setting up linear network....")
			self.model = Sequential()
			# self.model.add(Dense(env.action_space.n, input_dim = env.observation_space.shape[0], activation='linear', kernel_initializer='he_uniform', use_bias = True))
			self.model.add(Dense(32, input_dim = env.observation_space.shape[0], activation='linear', kernel_initializer='he_uniform',use_bias = True))
			self.model.add(Dense(env.action_space.n, input_dim = 32, activation='linear', kernel_initializer='he_uniform',use_bias = True))
			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='graphs/Linear.png', show_shapes = True)
			self.model.summary()

		elif(deep==True):
			print("Setting up DDQN network....")
			self.model = Sequential()
			self.model.add(Dense(32, input_dim = env.observation_space.shape[0], activation='relu', kernel_initializer='he_uniform',use_bias = True))
			# self.model.add(BatchNormalization())
			self.model.add(Dense(32, input_dim = 32,activation='relu', kernel_initializer='he_uniform',use_bias = True))
			# self.model.add(BatchNormalization())
			self.model.add(Dense(32, input_dim = 32, activation='relu', kernel_initializer='he_uniform',use_bias = True))
			# self.model.add(BatchNormalization())
			self.model.add(Dense(env.action_space.n, input_dim = 32, activation='linear', kernel_initializer='he_uniform',use_bias = True))
			print("Q-Network initialized.... :)\n")

			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='graphs/DDQN.png', show_shapes = True)
			self.model.summary()

		elif(duel==True):
			print("Setting up Dueling DDQN network....")
			inp = Input(shape=(env.observation_space.shape[0]+1,))
			layer_shared1 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
			# layer_shared1 = BatchNormalization()(layer_shared1)
			layer_shared2 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared1)
			layer_shared2 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
			# layer_shared2 = BatchNormalization()(layer_shared2)
			print("Shared layers initialized....")

			layer_v1 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
			# # layer_v1 = BatchNormalization()(layer_v1)
			layer_a1 = Dense(32,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
			# layer_a1 = BatchNormalization()(layer_a1)
			layer_v2 = Dense(1,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_v1)
			layer_a2 = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_a1)
			print("Value and Advantage Layers initialised....")

			layer_mean = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(layer_a2)
			temp = layer_v2
			temp2 = layer_mean

			for i in range(env.action_space.n-1):
				layer_v2 = keras.layers.concatenate([layer_v2,temp],axis=-1)
				layer_mean = keras.layers.concatenate([layer_mean,temp2],axis=-1)

			# layer_q = Lambda(lambda x: K.expand_dims(x[0],axis=-1)  + x[1] - K.mean(x[1],keepdims=True), output_shape=(env.action_space.n,))([layer_v2, layer_a2])
			layer_q = Subtract()([layer_a2,layer_mean])
			layer_q = Add()([layer_q,layer_v2])

			print("Q-function layer initialized.... :)\n")

			self.model = Model(inp, layer_q)
			self.model.summary()
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
		# self.model.load_weights(weight_file)
		self.model.set_weights(weight_file)

	def visualise_weights(self):
		print("Current Weights\n")
		for layer in self.model.layers:
			temp = layer.get_weights()



class DQN_Agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, env, replay, deep, duel, render, env_name):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.net = QNetwork(env,replay,deep,duel)
		self.prediction_net = QNetwork(env,replay,deep,duel)
		self.replay = replay
		self.deep = deep
		self.duel = duel
		self.env = env
		self.env_name = env_name
		self.render = render
		self.feature_size = env.observation_space.shape[0]
		self.action_size = env.action_space.n
		self.discount_factor = 1

		if(env_name == "CartPole-v0"):
			self.discount_factor = 0.99
		elif(env_name == "MountainCar-v0"):
			self.discount_factor = 1

		self.train_iters = 1000000
		self.epsilon = 0.5 																						#HYPERPARAMETER3
		self.epsilon_min = 0.05																					#HYPERPARAMETER4
		self.num_episodes = 4000
		self.epsilon_decay = float((self.epsilon-self.epsilon_min)/100000)										#HYPERPARAMETER5
		self.update_prediction_net_iters =500 																	#HYPERPARAMETER6
		self.avg_rew_buf_size_epi = 10
		self.save_weights_iters = 10000
		self.save_model_iters = 10000
		self.print_epi = 1
		self.print_loss_epi = 50
		self.main_goal = 0.5
		self.evaluate = 0.0

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		return np.argmax(q_values[0])

	def test(self, model_file, epi=200):

		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.net.model = keras.models.load_model(model_file)
		reward_array = np.zeros(epi)

		tot_reward = 0
		for e in range(epi):
			curr_reward = 0
			curr_state = self.env.reset()
			curr_action = np.random.randint(self.action_size)
			while(True):
				nextstate, reward, is_terminal, debug_info = self.env.step(curr_action)
				curr_reward += reward
				if(is_terminal):
					break
				nextstate = np.append(nextstate,self.main_goal)
				nextstate = nextstate.reshape([1,self.feature_size+1])
				q_nextstate = self.net.model.predict(nextstate)
				nextaction = self.greedy_policy(q_nextstate)

				curr_state = nextstate
				curr_action = nextaction

			tot_reward += curr_reward
			reward_array[e] = curr_reward

		self.evaluate = float(tot_reward/epi)

		return float(tot_reward/epi), reward_array

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=bool,default=False)
	parser.add_argument('--train',dest='train',type=bool,default=True)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--deep',dest='deep',type=bool,default=False)
	parser.add_argument('--duel',dest='duel',type=bool,default=False)
	parser.add_argument('--replay',dest='replay',type=bool,default=False)
	return parser.parse_args()

def plot_graph(env_name, rewards, type_plot, algo):
	# plt.plot(range(0,len([3,4])),[3,4])
	# plt.show()
	plt.cla()
	plt.plot(range(0,len(rewards)),rewards)
	imgname = 'temp.png'
	if(type_plot == 'train'):
		imgname = 'train.png'
	elif(type_plot == 'test'):
		imgname = 'test.png'

	if(env_name=="CartPole-v0"):
		if('deep'==algo):
			plt.savefig('plots/deep/cartpole/'+imgname)
		elif('duel'==algo):
			plt.savefig('plots/duel/cartpole/'+imgname)
		elif('replay'==algo):
			plt.savefig('plots/replay/cartpole/'+imgname)
		else:
			plt.savefig('plots/linear/cartpole/'+imgname)
	elif(env_name=="MountainCar-v0"):
		if('deep'==algo):
			plt.savefig('plots/deep/mountaincar/'+imgname)
		elif('duel'==algo):
			plt.savefig('plots/duel/mountaincar/'+imgname)
		elif('replay'==algo):
			plt.savefig('plots/replay/mountaincar/'+imgname)
		else:
			plt.savefig('plots/linear/mountaincar/'+imgname)

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

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

	agent = DQN_Agent(env,args.replay,args.deep,args.duel,args.render,args.env)

	folder = ''
	if(args.env=="MountainCar-v0"):
		if(args.deep==True):
			folder = 'deep/mountaincar'
		elif(args.duel==True):
			folder = 'duel/mountaincar'
		elif(args.replay==True):
			folder = 'replay/mountaincar'
		else:
			folder='linear/mountaincar'
	if(args.env=="CartPole-v0"):
		if(args.deep==True):
			folder = 'deep/cartpole'
		elif(args.duel==True):
			folder = 'duel/cartpole'
		elif(args.replay==True):
			folder = 'replay/cartpole'
		else:
			folder='linear/cartpole'


	final_weight_file = 'temp'
	print("Calculating test rewards...")
	test_files = []
	test_reward = []
	for filename in sorted_aphanumeric(os.listdir('models/'+folder)):
		if(filename!='.DS_Store' and isfile('models/'+join(folder,filename))):
			test_files.append(filename)

	for filename in test_files:
		print(filename)
		r,_ = agent.test('models/'+join(folder,filename),20)
		test_reward.append(r)
		final_weight_file = 'models/'+join(folder,filename)

	train_file = ''

	if(environment_name=="CartPole-v0"):
		if(True==args.deep):
				train_file = 'data/train_rew_cp_duel.pkl'
		elif(True==args.duel):
				train_file = 'data/train_rew_cp_duel.pkl'
		elif(True==args.replay):
				train_file = 'data/train_rew_cp_duel.pkl'
		else:
				train_file = 'data/train_rew_cp_duel.pkl'
	elif(environment_name=="MountainCar-v0"):
		if(True==args.deep):
				train_file = 'data/train_rew_cp_duel.pkl'
		elif(True==args.duel):
				train_file = 'data/train_rew_cp_duel.pkl'
		elif(True==args.replay):
				train_file = 'data/train_rew_cp_duel.pkl'
		else:
				train_file = 'data/train_rew_cp_duel.pkl'

	print("Starting to plot..")
	#plot discounted training and undiscounted testing rewards at intervals
	if(args.deep==True):
		# plot_graph(args.env,train_reward,'train','deep')
		plot_graph(args.env,test_reward,'test','deep')
	elif(args.duel==True):
		# plot_graph(args.env,train_reward,'train','duel')
		plot_graph(args.env,test_reward,'test','duel')
	elif(args.replay==True):
		# plot_graph(args.env,train_reward,'train','replay')
		plot_graph(args.env,test_reward,'test','replay')
	else:
		# plot_graph(args.env,train_reward,'train','linear')
		plot_graph(args.env,test_reward,'test','linear')

	print("Plot done")

	#final test reward gets saves in .txt
	print("Saving mean and std..")
	_, t = agent.test(final_weight_file)

	file_test = open('final_test.txt', 'a')

	if(args.deep==True):
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-deep-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-deep-mean\t"+str(np.mean(t))+"\n")
		elif(args.env=="CartPole-v0"):
			file_test.write("cartpole-deep-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-deep-mean\t"+str(np.mean(t))+"\n")
	if(args.duel==True):
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-duel-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-duel-mean\t"+str(np.mean(t))+"\n")
		elif(args.env=="CartPole-v0"):
			file_test.write("cartpole-duel-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-duel-mean\t"+str(np.mean(t))+"\n")
	if(args.replay==True):
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-replay-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-replay-mean\t"+str(np.mean(t))+"\n")
		elif(args.env=="CartPole-v0"):
			file_test.write("cartpole-replay-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-replay-mean\t"+str(np.mean(t))+"\n")
	else:
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-linear-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-linear-mean\t"+str(np.mean(t))+"\n")
		elif(args.env=="CartPole-v0"):
			file_test.write("cartpole-linear-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-linear-mean\t"+str(np.mean(t))+"\n")

if __name__ == '__main__':
	main(sys.argv)
