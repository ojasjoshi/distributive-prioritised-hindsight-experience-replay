from __future__ import division
from collections import deque
import os
import warnings

import random
import numpy as np
import keras.backend as K
import keras.optimizers as optimizers
from copy import deepcopy

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


# Deep DPG as described by Lillicrap et al. (2015)
# http://arxiv.org/pdf/1509.02971v2.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4324&rep=rep1&type=pdf
class DDPGAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, do_HER=True, K=4, HER_strategy='future',
                 do_PER=True, epsilon = 1e-4, **kwargs):
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError('Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError('Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.format(critic))
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated action input "{}".'.format(critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError('Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update  
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects # (custom objectives)

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(critic_action_input)     # (index of the input layer during new batch)(?)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()
        self.current_episode_experience = []    # (stores current episode experience as [s',a, s, r, is_terminal])
        self.K = K 
        self.do_HER = do_HER
        self.do_PER = do_PER
        self.strategy = HER_strategy
        ## (TODO: try to take it into memory.py/PER)
        self.epsilon = epsilon

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError('More than optimizers provided. Please only provide a maximum of two optimizers, the first one for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimzer and loss
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            critic_updates = get_soft_target_model_updates(self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer, loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        # Assuming critic's state inputs are the same as actor's.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append([])
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_inputs[self.critic_action_input_idx] = self.actor(critic_inputs)

        combined_output = self.critic(combined_inputs)

        updates = actor_optimizer.get_updates(
            params=self.actor.trainable_weights, loss=-K.mean(combined_output))  # (gradient updates for actor)
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(self.target_actor, self.actor, self.target_model_update)
        updates += self.actor.updates  # include other updates of the actor, e.g. for BN

        # Finally, combine it all into a callable function.
        if K.backend() == 'tensorflow':
            self.actor_train_fn = K.function(critic_inputs + [K.learning_phase()],          # (function to apply gradient updates)
                                             [self.actor(critic_inputs)], updates=updates)
        else:
            if self.uses_learning_phase:
                critic_inputs += [K.learning_phase()]
            self.actor_train_fn = K.function(critic_inputs, [self.actor(critic_inputs)], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:   # (for exploration)
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, nextstate, info, env, terminal=False):
        # Store most recent experience in memory.

        """ WARNING: Keep self.memory_interval=1"""
        assert self.memory_interval == 1

        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, nextstate, reward, terminal,
                               training=self.training)       

            self.current_episode_experience.append([self.recent_observation, self.recent_action, nextstate, reward, terminal, info])
        
        if(terminal==True):
            if(self.do_HER):
                self.add_HER(env,strategy=self.strategy)
            self.current_episode_experience = []    # reset current episode experience if new episode begins

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor # (check warmup is over or not and can both actor and critic can be trained)
        if can_train_either and self.step % self.train_interval == 0:
            if(self.do_PER):
                experiences, experience_idxs = self.memory.sample(batch_size=self.batch_size)                   # (samples the batch from replay)
                assert len(experiences) == len(experience_idxs)
            else:
                experiences = self.memory.sample(batch_size=self.batch_size)                   # (samples the batch from replay)
                
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation). # (vectorized => list)
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            if(self.do_PER):
                weights_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)
                if(self.do_PER):
                    weights_batch.append(e.importance_weight)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            if(self.do_PER):
                weights_batch = np.array(weights_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)
            if(self.do_PER):
                assert weights_batch.shape == (self.batch_size,)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:

                if(self.do_PER):
                    # Find input q_values
                    input_actions = action_batch
                    assert input_actions.shape == (self.batch_size, self.nb_actions)
                    if len(self.critic.inputs) >= 3:
                        state0_batch_with_action = state0_batch[:]
                    else:
                        state0_batch_with_action = [state0_batch]
                    state0_batch_with_action.insert(self.critic_action_input_idx, input_actions)
                    """ IMPORTANT: input_q_values predicted on critic network not target_critic """
                    input_q_values = self.critic.predict_on_batch(state0_batch_with_action).flatten()
                    assert input_q_values.shape == (self.batch_size,)

                # Find target q_values
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                if(self.do_PER):
                    """ Prioritised Experience Replay Implementation """
                    # update the priorities of transitions with TD errors
                    TD_errors = abs(targets - np.expand_dims(input_q_values,axis=1)).flatten() + self.epsilon*np.ones(shape=(self.batch_size,))
                    self.memory.update_priorities(TD_errors, experience_idxs)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                
                if(self.do_PER):
                    """" CHECK: sample_weight usage """
                    metrics = self.critic.train_on_batch(state0_batch_with_action, targets, sample_weight=weights_batch)
                else:
                    metrics = self.critic.train_on_batch(state0_batch_with_action, targets)

                if self.processor is not None:
                    metrics += self.processor.metrics
            
            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)
                
        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics


    ## FetchSlide/FetchPush: [3:6] is the achieved goal, [-3:] is the desired goal (when wrapper.flatten is used w/o 'achieved goal' as key)
    ## FetchReach: [0:3] is the achieved goal, [-3:] is the desired goal (when wrapper.flatten is used w/o 'achieved goal' as key)

    def add_HER(self, env, strategy='future'):
        for t in range(len(self.current_episode_experience)):
            sample_index = t
            if(strategy=='episode'):
                sample_index = 0
            for k in range(self.K):
                """ random.randint is too slow check: (https://www.reddit.com/r/Python/comments/jn0bb/randomrandint_vs_randomrandom_why_is_one_15x/) """
                # new_goal_idx = np.random.randint(sample_index,len(self.current_episode_experience))
                new_goal_idx = random.sample(range(sample_index,len(self.current_episode_experience)),1)[0]
                new_goal = deepcopy(self.current_episode_experience[new_goal_idx][2][3:6])    # randomly sampled substitute goal from states seen after the current transition
                
                # update the original transition
                her_curr_observation, curr_action, her_next_observation, _ , curr_terminal, info = self.current_episode_experience[t]
                
                her_achieved_goal = deepcopy(her_next_observation[3:6])
                her_next_observation = deepcopy(her_next_observation)
                her_curr_observation = deepcopy(her_curr_observation)

                her_next_observation[-3:] = new_goal 
                her_curr_observation[-3:] = new_goal 
                her_reward = env.compute_reward(her_achieved_goal, new_goal, info)

                """ WARNING: NonSequentialMemory required """
                self.memory.append(her_curr_observation, curr_action, her_next_observation, her_reward, curr_terminal)


