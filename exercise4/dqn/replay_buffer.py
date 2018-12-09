from collections import namedtuple
import numpy as np
import os
import gzip
import pickle

class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, size=1e5):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.buffer_size = size
        self.buffer_index = 0

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        If the buffer is full, first added actions will be deleted.
        """
        if(len(self._data.states) < (self.buffer_size + 1)):
            self._data.states.append(state)
            self._data.actions.append(action)
            self._data.next_states.append(next_state)
            self._data.rewards.append(reward)
            self._data.dones.append(done)
        else:
            index = int(self.buffer_index)
            self._data.states[index] = state
            self._data.actions[index] = action
            self._data.next_states[index] = next_state
            self._data.rewards[index] = reward
            self._data.dones[index] = done
            self.buffer_index += 1
            self.buffer_index %= self.buffer_size
        #self._check_capacity()

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

    # checks replay buffer for free capacity to add new items
    def _check_capacity(self):
        num_states = len(self._data.states)
        if num_states > self.buffer_size:
            num_delete = int(num_states - self.buffer_size)
            self._data.states = self._data.states[num_delete:]
            self._data.actions = self._data.actions[num_delete:]
            self._data.next_states = self._data.next_states[num_delete:]
            self._data.rewards = self._data.rewards[num_delete:]
            self._data.dones = self._data.dones[num_delete:]
