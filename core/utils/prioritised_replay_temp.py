class MemoryRecord(object):
    def __init__(self, transition_list=[], game_over=False, transition_powered_priority=1):
        self.transition_list = transition_list
        self.transition_powered_priority = transition_powered_priority
        self.game_over = game_over
        self.is_closed = False

    def add_transition(self, transition, game_over, transition_powered_priority=1):
        # add a single transition to a record and update the game over state
        if self.is_closed:
            raise Exception('record finalized')
        self.transition_list += [transition]
        self.transition_powered_priority = transition_powered_priority
        self.game_over = game_over

    def finalize(self):
        # finalize the record so no more transitions will be added
        self.is_closed = True
        

class ExperienceReplay(object):
    # memory consists of tuples [transition, game_over, priority^alpha]
    def __init__(self, max_memory=50000, prioritized=False, store_episodes=False):
        # experience replay structure params
        self.max_memory = max_memory
        self.memory = []
        self.store_episodes = store_episodes

        # prioritized experience replay params
        self.prioritized = prioritized
        self.alpha = 0.6 # prioritization factor
        self.beta_start = 0.4
        self.beta_end = 1
        self.beta = self.beta_end
        self.sum_powered_priorities = 0 # sum p^alpha

    def is_last_record_closed(self):
        return self.memory == [] or self.memory[-1].is_closed == True

    def add_record(self, transition, game_over, transition_powered_priority):
        record = MemoryRecord([transition], game_over, transition_powered_priority)
        self.memory.append(record)

    def get_last_record(self):
        return self.memory[-1]

    def close_last_record(self):
        self.get_last_record().finalize()

    def remember(self, transition, game_over):
        """Add a transition to the experience replay
        :param transition: the transition to insert
        :param game_over: is the next state a terminal state?
        """
        # set the priority to the maximum current priority
        transition_powered_priority = 1e-7 ** self.alpha
        if self.prioritized:
            transition_powered_priority = np.max(self.memory,1)[0,2]
        self.sum_powered_priorities += transition_powered_priority

        # store transition
        if self.is_last_record_closed():
            self.add_record(transition, game_over, transition_powered_priority)
        else:
            self.get_last_record().add_transition(transition, game_over, transition_powered_priority)
        # finalize the record if necessary
        if not self.store_episodes or (self.store_episodes and (game_over or transition.reward > 0)): #TODO: this is wrong
            self.close_last_record()

        # free some space (delete the oldest transition or episode)
        if len(self.memory) > self.max_memory:
            if self.prioritized:
                if self.store_episodes:
                    self.sum_powered_priorities -= np.sum(np.array(self.memory)[0,:,2])
                else:
                    self.sum_powered_priorities -= self.memory[0].transition_powered_priority
            del self.memory[0]

    def sample_minibatch(self, batch_size, not_terminals=False):
        """Samples one minibatch of transitions from the experience replay
        :param batch_size: the minibatch size
        :param not_terminals: sample or don't sample transitions were the next state is a terminal state
        :return: a list of tuples of the form: [idx, transition, game_over, weight]
        """
        batch_size = min(len(self.memory), batch_size)
        if self.prioritized: # TODO: not currently working for episodic experience replay
            # prioritized experience replay
            probs = np.random.rand(batch_size)
            importances = [self.get_transition_importance(idx) for idx in range(len(self.memory))]
            thresholds = np.cumsum(importances)

            # multinomial sampling according to priorities
            indices = []
            for p in probs:
                for idx, threshold in zip(range(len(thresholds)), thresholds):
                    if p < threshold:
                        indices += [idx]
                        break
        else:
            indices = np.random.choice(len(self.memory), batch_size)

        # TODO: this is just a simple test
        #positives = [idx for idx, transition in enumerate(self.memory) if self.memory[idx].transition_list[-1].reward > 0]
        #print(positives)
        #print(np.random.choice(positives, batch_size/2))
        #print(indices)
        #indices = np.append(indices, np.random.choice(positives, batch_size/2))
        #print(indices)

        minibatch = list()
        for idx in indices:
            while not_terminals and self.memory[idx].game_over:
                idx = np.random.choice(len(self.memory), 1)[0]
            weight = 0
            if self.prioritized: # TODO: not working for episodic experience replay
                weight = self.get_transition_weight(idx)
            minibatch.append([idx, self.memory[idx].transition_list, self.memory[idx].game_over, weight])  # idx, [transition, transition, ...] , game_over, weight

        if self.prioritized:
            max_weight = np.max(minibatch,0)[3]
            for idx in range(len(minibatch)):
                minibatch[idx][3] /= float(max_weight) # normalize weights relative to the minibatch

        #print([record[0] for record in minibatch])
        #print([idx for idx, transition in enumerate(self.memory) if self.memory[idx].transition_list[-1].reward > 0])
        #print(minibatch)
        return minibatch

    def update_transition_priority(self, transition_idx, priority):
        """Update the priority of a transition by its index
        :param transition_idx: the index of the transition
        :param priority: the new priority
        """
        self.sum_powered_priorities -= self.memory[transition_idx].transition_powered_priority
        powered_priority = (priority+np.spacing(0)) ** self.alpha
        self.sum_powered_priorities += powered_priority
        self.memory[transition_idx].transition_powered_priority = powered_priority

    def get_transition_importance(self, transition_idx):
        """Get the importance of a transition by its index
        :param transition_idx: the index of the transition
        :return: the importance - priority^alpha/sum(priority^alpha)
        """
        powered_priority = self.memory[transition_idx].transition_powered_priority
        importance = powered_priority / float(self.sum_powered_priorities)
        return importance

    def get_transition_weight(self, transition_idx):
        """Get the weight of a transition by its index
        :param transition_idx: the index of the transition
        :return: the weight of the transition - 1/(importance*N)^beta
        """
        weight = 1/float(self.get_transition_importance(transition_idx)*self.max_memory)**self.beta
        return weight