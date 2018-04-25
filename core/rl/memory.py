from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
from queue import PriorityQueue
from copy import deepcopy
import heapq

import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
PrioritisedExperience = namedtuple('PrioritisedExperience', 'state0, action, reward, state1, terminal1, importance_weight')


def sample_batch_indexes(low, high, size):
    """Return a sample of (size) unique elements between low and high

        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick

        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return element of buffer at specific index

        # Argument
            idx (int): Index wanted

        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        """Append an element to the buffer

        # Argument
            v (object): Element to append
        """
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations

        # Argument
            current_observation (object): Last observation

        # Returns
            A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory
        
        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]      #choose random index in this case (do it till the transition is not terminal)
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

## OJAS Implementation
class NonSequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(NonSequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations0 = RingBuffer(limit)
        self.observations1 = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:

            state0 = [self.observations0[idx - 1]]      # (idx-1 or idx)?!
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]
            state1 = [self.observations1[idx - 1]]
            while len(state1) < self.window_length:
                state1.insert(0, zeroed_observation(state1[0]))

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation0, action, observation1, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super(NonSequentialMemory, self).append(observation0, action, reward, terminal, training=training)

        # This needs to be understood as follows: from `observation0`, take `action`, obtain `reward`+'observation1'
        # and weather the observation1 is `terminal` or not.
        if training:
            self.observations0.append(observation0)
            self.actions.append(action)
            self.observations1.append(observation1)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.observations0)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(NonSequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

## OJAS Implementation
## TODO: Update as a priority queue to remove elements in priority + implement rank-based method
""" TODO: Check use of self.length vs len(self.heap) """
class PrioritisedRingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.length = 0
        self.heap_data = []
        self.overflow = 0 # indicates the first time the memory overflows

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return element of buffer at specific index

        # Argument
            idx (int): Index wanted

        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()

        return self.heap_data[idx]

    def append(self, v):
        """Append an element to the buffer. If full remove element with least priority.

        # Argument
            v (object): Element to append
        """
        
        """ TODO: remove if no assertion error in future """
        assert len(self.heap_data)==self.length
        if(self.overflow==1):                   # to check if after full memory, it stays full
            assert self.length == self.maxlen

        if self.length < self.maxlen:
            # We have space, simply increase the length.
            heapq.heappush(self.heap_data, v)
            self.length += 1
            
            return None

        elif self.length == self.maxlen:
            # First time the memory is full
            if(self.overflow==0):
                self.overflow = 1
                print("\nMemory full. Running time peaked!!")
            # No space, "remove" the first item.
            removed = heapq.heapreplace(self.heap_data, v)

            return removed[0]
        else:
            # This should never happen.
            raise RuntimeError()

    def update_heap_priorities(self, new_priorities, idxs):
        """ Input: new_priorities- list of new priorities, idxs- list of indexes to update (called after a batch update (bigger batch size better) )"""

        for (new_val,idx) in zip(new_priorities, idxs):
            self.heap_data[idx][0] = float(new_val)
            self.heap_data[idx][1].priority = float(new_val)
        # now update the heap structure 
        heapq.heapify(self.heap_data)

## OJAS Implementation
class PrioritisedNonSequentialMemory(Memory):
    def __init__(self, limit=1000000, alpha=0.7, beta=0.5, **kwargs):
        super(PrioritisedNonSequentialMemory, self).__init__(**kwargs)
        self.limit = limit
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.powered_priority_sum = 0.0
        self.max_imp_weight = 1.0
        self.data = PrioritisedRingBuffer(limit) # (elements are [priority,Transition object] )

    def update_priorities(self, new_priorities, idxs):
        """ Input-
        new_priorities: list/np.array of new priorities
        idxs: list/np.array of indexes to update """
        
        """ WARNING: mainatain calling order """
        self.update_PoweredSum(new_priorities, idxs)
        self.data.update_heap_priorities(new_priorities, idxs)
        self.update_MaxPriorityandWeight()

    """ BOTTLENECK WARNING: O(n) [called after batch update => keep larger batchsize] """
    def update_MaxPriorityandWeight(self):
        """ Update the maximum_priority and maximum_importance_weight """

        self.max_priority = float(heapq.nlargest(1, self.data.heap_data)[0][0])
        self.max_imp_weight = deepcopy(self.get_importance_weight(self.max_priority))

    """ DEPRECATED USE """
    def get_old_priorities(self, idxs):
        return [float(self.data[idx][0]) for idx in idxs]

    def update_PoweredSum(self, new_priorities, idxs):
        """ Update powered_sum with new batch updates """

        for (new_val,idx) in zip(new_priorities, idxs):
            self.powered_priority_sum -= float(self.data.heap_data[idx][0])**self.alpha
            self.powered_priority_sum += float(new_val)**self.alpha

    def append(self, state0, action, state1, reward, terminal, training=True):
        """ TODO: edit last_id, update transition_sampledornot"""
        removed_priority = self.data.append([self.max_priority, Transition(state0, action, reward, state1, terminal, self.max_priority)])
        
        if(removed_priority!=None):
            self.powered_priority_sum -= float(removed_priority)**self.alpha

        self.powered_priority_sum += float(self.max_priority)**self.alpha


    """ WARNING: problem suppressed (ValueError: probabilities do not sum to 1) by normalising """
    def get_probabilites_from_index(self, low, high):
        """ utility function for sample_batch_indexes """
        """ returns probabilities in sequence stored that in memory """
        BAD_PROB_THRESHOLD = 1e-2
        prob_list = []
        
        for idx in range(low,high):
            prob_list.append((float(self.data[idx][0])**self.alpha)/float(self.powered_priority_sum))

        normalizer = np.sum(prob_list)

        assert abs(normalizer-1.0) < BAD_PROB_THRESHOLD
        
        return prob_list/normalizer

    """ BOTTLENECK WARNING: np.random.choice """
    def sample_batch_indexes(self, low, high, size):
        if high - low >= size:
            try:
                r = xrange(low, high)
            except NameError:
                r = range(low, high)

            batch_idxs = np.random.choice(r, size, p=self.get_probabilites_from_index(low,high), replace=False)
        else:
            # Not enough data. Help ourselves with sampling from the range, but the same index
            # can occur multiple times. This is not good and should be avoided by picking a
            # large enough warm-up phase. Not a prioritised batch in this case. (oversampling)
            warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling! Current batch not sampled with priority.')
            batch_idxs = np.random.random_integers(low, high - 1, size=size)
        assert len(batch_idxs) == size
        return batch_idxs

    """ BOTTLENECK WARNING: O(n) [called after batch update => keep larger batchsize]"""
    def sample(self, batch_size=32, batch_idxs=None):
        """TODO: Update transitions change to update idxs """
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:

            batch_idxs = self.sample_batch_indexes(
                0, self.nb_entries, size=batch_size)

        assert np.min(batch_idxs) >= 0
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:

            state0 = [self.data[idx][1].transition_data['state0']]
            action = self.data[idx][1].transition_data['action']
            reward = self.data[idx][1].transition_data['reward']
            terminal1 = self.data[idx][1].transition_data['terminal']
            state1 = [self.data[idx][1].transition_data['state1']]

            assert len(state0) == 1
            assert len(state1) == len(state0)

            """ Calculate importance weights """
            imp_weight = self.get_importance_weight(self.data[idx][1].priority)

            experiences.append(PrioritisedExperience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1, importance_weight=imp_weight))

        assert len(experiences) == batch_size

        return experiences, batch_idxs
    
    def update_memory_if_append(self):
        raise NotImplementedError()

    def get_proportional_priority(self, priority):
        """ Returns proportional priority given priority"""
        return float(float(priority)**self.alpha) / float(self.powered_priority_sum)

    def get_importance_weight(self, priority):
        """ Returns the importance weight given priority"""
        weight = (1/float(self.max_imp_weight))*(1/float(self.get_proportional_priority(priority)*self.limit)**self.beta)
        return weight

    # @classmethod
    def update_memory_if_update(self, old_priority, new_priority):
        raise NotImplementedError()

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.data)

    """TODO: add more config values """
    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(PrioritisedNonSequentialMemory, self).get_config()
        config['limit'] = self.limit
        config['alpha'] = self.alpha
        config['beta'] = self.beta

        return config

## OJAS Implementation
""" Utility class for PrioritisedNonSequentialMemory """
class Transition(object):
    def __init__(self, state0, action, reward, state1, terminal, priority, prop_priority=1, **kwargs):
        self.transition_data = {'state0':state0, 'action':action, 'reward':reward, 'state1':state1, 'terminal':terminal}
        self.priority = priority
        self.sampled = False

    def update_priority(self, new_priority, memory):
        """ DEPRECATED USE """
        # old_priority = deepcopy(self.priority)
        # self.priority = new_priority
        raise NotImplementedError()
        # self.proportional_priority = PrioritisedNonSequentialMemory.get_proportional_priority(new_priority) ## WARNING: get_proportional_prioirty is not a @classmethod now (cannot uncomment without changes in PrioritisedNonSequentialMemory)

    def __lt__(self, other):
        return self.transition_data['reward'] < other.transition_data['reward']

    def __le__(self, other):
        return self.transition_data['reward'] <= other.transition_data['reward']

    def __eq__(self, other):
        return self.transition_data['reward'] == other.transition_data['reward']

    def __ne__(self, other):
        return self.transition_data['reward'] != other.transition_data['reward']

    def __gt__(self, other):
        return self.transition_data['reward'] > other.transition_data['reward']

    def __ge__(self, other):
        return self.transition_data['reward'] >= other.transition_data['reward']


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of params and rewards

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of params randomly selected and a list of associated rewards
        """
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        """Append a reward to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        """Return number of episode rewards

        # Returns
            Number of episode rewards
        """
        return len(self.total_rewards)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config
