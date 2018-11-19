""" fake.py contains an OpenAI gym interface, used to temporarily fake
    interactions with ROS/Gazebo and the strawberry location prediction network.

    Author: Jonathon Sather
    Last updated: 2/18/2018
"""

import gym
from gym import wrappers

last_reward = None
last_terminal = None
last_state = None

class fake_plant:
    """ Fake plant object which acts as placeholder for interfacting with
        strawberry plant through ROS/Gazebo.
    """

    def __init__(self):
        pass

    def new(self):
        pass

class fake_agent:
    """ Fake agent object which acts as placeholder for interfacing with
        harvester agent through ROS/Gazebo. Uses OpenAI gym to simulate steps
        in the environment.
    """

    def __init__(self):
        """ Initializes fake ROS/Gazebo agent. Starts Pendulum environment from
            OpenAI gym.
        """

        self.env = gym.make('MountainCarContinuous-v0')#'Pendulum-v0')
        self.env.seed(0)

        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high

    def get_state_shape(self):
        """ Returns state shape. """

        return self.state_shape

    def get_action_shape(self):
        """ Returns action shape. """

        return self.action_shape

    def get_action_bound(self):
        """ Returns action bound. """

        return self.action_bound

    def get_obs(self):
        """ Returns last state. """
        global last_state

        joint_angles = []

        return last_state, joint_angles

    def reset(self):
        """ Resets gym environment. Simulates resetting harvesting arm to
            starting position.
        """
        global last_state

        last_state = self.env.reset()

    def step(self, action):
        """ Takes step in gym environment and returns observation. Simulates
            taking action with harvesting arm and getting observation and joint
            state.
        """
        global last_reward, last_terminal

        self.env.render()

        state, last_reward, last_terminal, _ = self.env.step(action)
        joint_angles = []

        return state, joint_angles

class fake_predictor:
    """ Fake predictor object which acts as a placeholder for interfacing with
        strawberry pose predictor. Uses global variables output from OpenAI gym
        to give "fake" responses.
    """

    def __init__(self):
        pass

    def predict(self, observation):
        """ Makes fake prediction given observation. Returns fake confidence
            value and terminal state flag from gym environment.
        """

        global last_terminal

        return 0.8, last_terminal

    def get_reward(self, confidence, terminal, action):
        """ Simulates getting reward given confidence, terminal flag, and
            action. Actually just returns reward from last gym call.
        """

        global last_reward

        return float(last_reward)

class fake_latent:
    """ Fake latent network which pretends to learns low-dimensional
        representation of camera images.
    """

    def __init__(self):
        pass

    def convert(self, observation):
        pass
