import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym.wrappers import Monitor

model_path = str(sys.argv[1])

def huber_loss_custom(y_true,y_pred):
    return tf.losses.huber_loss(y_true,y_pred)
    
def greedy_policy(q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values[0])

def boltzman_policy(q_values,tau=1., clip=(-500., 500.)):
    q_values[0] = q_values[0].astype('float64')
    nb_actions = q_values[0].shape[0]

    exp_values = np.exp(np.clip(q_values[0] / tau, clip[0], clip[1]))
    probs = exp_values / np.sum(exp_values)
    action = np.random.choice(range(nb_actions), p=probs)
    return action

def render_one_episode(model,env,epi_num):
        rewards = []
        state = env.reset()
        state = state.reshape([1,env.observation_space.shape[0]])
        action = boltzman_policy(model.predict(state))
        while(True):
            env.render()
            # print(state)
            nextstate, reward, is_terminal, _ = env.step(action)
            rewards.append(reward)
            if(is_terminal == True):
                break                  
            state = nextstate.reshape([1,env.observation_space.shape[0]])
            action = boltzman_policy(model.predict(state))
        print("Total reward for {} : {}".format(epi_num,np.sum(rewards)))

def main():
    env = gym.make('LunarLander-v2')

    num_episodes=10
    # save_episode_id=np.around(np.linspace(0,num_episodes,num=1))
    # env = Monitor(env,'videos/A2C/1',video_callable= lambda episode_id: episode_id in save_episode_id, force=True)

    # model = keras.models.load_model(model_path)
    model = keras.models.load_model(model_path,custom_objects={'huber_loss_custom':huber_loss_custom})
    for i in range(num_episodes):
    	render_one_episode(model,env,i)

if __name__ == '__main__':
	main()


