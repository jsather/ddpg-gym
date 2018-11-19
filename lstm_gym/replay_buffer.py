""" replay_buffer.py contains the class ReplayBuffer, which can be used to
    create an experience replay for Q-learning.

    Author: Jonathon Sather
    Last updated: 2/23/2018

    Implementation inspired by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg
"""

from collections import deque
import random
import numpy as np
import pdb

class EpisodeReplayBuffer(object):
    """ Replay buffer object used for training recurrent neural network. Stores
        'buffer_size' trajectories with methods to add and remove elements at
        specified trace lengths.
    """

    def __init__(self, buffer_size, init_data=[]):
        """ Initialize experience replay buffer. """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(init_data)

    def add(self, episode):
        """ Adds a trajectory from one episode to the buffer. Removes oldest
            trajectory if buffer is full.
        """
        if self.count < self.buffer_size:
            self.buffer.append(episode)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(episode)

    def size(self):
        """ Returns number of elements (episodes) stored in replay buffer. """
        return self.count

    def sample_batch(self, batch_size, trace_length):
        """ Samples from 'batch_size' trajectories to create an array of traces
            of length 'trace_length'.
        """

        batch = []

        if self.count < batch_size:
            episodes = random.sample(self.buffer, self.count)
        else:
            episodes = random.sample(self.buffer, batch_size)

        for ep in episodes:
            start = np.random.randint(0, len(ep)+1-trace_length)
            batch.append(ep[start:start+trace_length])

        batch = np.reshape(np.array(batch), [batch_size*trace_length, -1])

        s_batch = np.vstack(batch[:,0])
        a_batch = np.vstack(batch[:,1])
        t_batch = np.vstack(batch[:,2])
        r_batch = np.vstack(batch[:,3])
        s2_batch = np.vstack(batch[:,4])

        return (s_batch, a_batch, t_batch, r_batch, s2_batch)

    def clear(self):
        """ Clears replay buffer and resets count. """
        self.buffer.clear()
        self.count = 0

class ReplayBuffer(object):
    """ Replay buffer object used for training neural network. Stores
       'buffer_size' elements in buffer with methods to add and remove elements.
    """

    def __init__(self, buffer_size, init_data=[]):
        """ Initialize experience replay buffer. """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(init_data)

    def add(self, s, a, r, t, s2):
        """ Adds one state-transition tuple to the buffer. Removes oldest tuple
            if buffer is full.
        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """ Returns number of elements in replay buffer. """
        return self.count

    def sample_batch(self, batch_size):
        """ Samples 'batch_size' number of state-transition tuples from replay
            buffer. If the replay buffer has less than 'batch_size' elements,
            return all elements.
        """
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        """ Clears replay buffer and resets count. """
        self.buffer.clear()
        self.count = 0
