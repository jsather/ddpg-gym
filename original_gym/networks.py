""" networks.py contains the classes for defining the actor and critic networks
    used in the DDPG algorithm.

    Author: Jonathon Sather
    Last updated: 2/23/2018

    Uses parameters from original DDPG paper:
    'Continuous Control in Deep Reinforcement Learning' by Lillicrap, et al.
    (arXiv:1509.02971)

    Influenced by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg

    TODO: 1. Add recurrence units.
"""

import tensorflow as tf
import tflearn
from collections import deque
import random
import numpy as np

from noise import OrnsteinUhlenbeckActionNoise

#--------------- Helper functions ---------------#
def sqrt_unif_init(n):
    """ Creates bounded uniform weight initializer with range +-1/sqrt(n). """
    return tf.random_uniform_initializer(-1/np.sqrt(n), 1/np.sqrt(n))

def dense_relu(x, size, fan_in, phase, l2_scale=0.0, scope='layer',
    batch_norm=True):
    """ Creates fully connected layer with uniform initialization. Includes
        batch normalization by default.
    """
    with tf.variable_scope(scope):
        h = tf.layers.dense(x, size, activation=None,
            kernel_initializer=sqrt_unif_init(fan_in),
            bias_initializer=sqrt_unif_init(fan_in),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_scale),
            name='dense')
        if batch_norm:
            h = tf.contrib.layers.batch_norm(h, scale=True, is_training=phase,
                updates_collections=None, scope='bn')
        return tf.nn.relu(h)

def dense_relu_2input(x1, x2, size, fan_in, l2_scale=0.0, scope='layer'):
    """ Creates dual input fully connected layer with uniform initialization.
        Does not include batch normalization.
    """
    with tf.variable_scope(scope):
        init_mag = 1/np.sqrt(fan_in)
        _, num_x1 = x1.get_shape().as_list()
        _, num_x2 = x2.get_shape().as_list()

        w1 = tf.Variable(tf.random_uniform([num_x1, size],
            minval=-init_mag, maxval=init_mag))
        w2 = tf.Variable(tf.random_uniform([num_x2, size],
            minval=-init_mag, maxval=init_mag))
        b = tf.Variable(tf.random_uniform([size],
            minval=-init_mag, maxval=init_mag))

        l2_reg = tf.contrib.layers.l2_regularizer(l2_scale)
        tf.contrib.layers.apply_regularization(l2_reg, weights_list=[w1, w2])

        join = tf.matmul(x1, w1) + tf.matmul(x2, w2) + b

        return tf.nn.relu(join)

def input_layer(size, phase, scope='input', batch_norm=True):
    """ Creates input placeholder and returns input to first hidden layer.
        Includes batch normalization by default.
    """
    with tf.variable_scope(scope):
        inputs = tf.placeholder(shape=[None, size], dtype=tf.float32,
            name='input')
        if batch_norm:
            to_h = tf.contrib.layers.batch_norm(inputs, scale=True,
                is_training=phase, updates_collections=None, scope='bn')
        else:
            to_h = inputs

        return inputs, to_h

def output_layer(x, size, unif_mag, act=None, scope='output'):
    """ Creates output layer with tanh nonlinearity and uniform initialization.
    """
    with tf.variable_scope(scope):
        out = tf.layers.dense(x, size, activation=act,
            kernel_initializer=tf.random_uniform_initializer(
                -unif_mag, unif_mag),
            bias_initializer=tf.random_uniform_initializer(-unif_mag, unif_mag),
            name='out')

        return out

#--------------- Network classes ---------------#
class ActorNetwork(object):
    """ Defines the actor network (Q-network optimizer) used in the DDPG
        algorithm.
    """
    def __init__(self, session, state_shape, action_shape, action_bound,
                 learning_rate, tau, batch_size):
        """ Initialize actor and target networks and update methods. """
        self.session = session
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Initialize addititve Ornstein Uhlenbeck noise
        self.OU_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_shape))

        # Initialize actor network
        self.phase = tf.placeholder(tf.bool, name='phase_act')
        self.inputs, self.out, self.scaled_out = \
            self.create_actor_network(self.phase)
        self.network_params = tf.trainable_variables()

        # Initialize target actor network
        self.target_inputs, self.target_out, self.target_scaled_out = \
            self.create_actor_network(self.phase, prefix='tar_')
        self.target_network_params = \
            tf.trainable_variables()[len(self.network_params):]

        # Define target update op
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) +
            tf.multiply(self.target_network_params[i], 1.0 - self.tau))
            for i in range(len(self.target_network_params))]

        # Define ops for getting necessary gradients
        self.action_gradient = \
            tf.placeholder(tf.float32, [None, self.action_shape])
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        # Define optimization op
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = \
            len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, phase, prefix=''):
        """ Constructs the actor network. Phase is boolean tensor corresponding
            to whether in training or testing phase.
        """
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        OUT_INIT_MAG = 3e-3

        # Construct network using convenience functions defined above
        inputs, to_fc1 = input_layer(self.state_shape, phase,
            scope=prefix+'in_act')
        fc1 = dense_relu(to_fc1, N_HIDDEN_1, self.state_shape, phase,
            scope=prefix+'fc1_act')
        fc2 = dense_relu(fc1, N_HIDDEN_2, N_HIDDEN_1, phase,
            scope=prefix+'fc2_act')
        out = output_layer(fc2, self.action_shape, OUT_INIT_MAG, act=tf.tanh,
            scope=prefix+'out_act')
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, action_gradient):
        """ Runs training step on actor network. """
        self.session.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: action_gradient,
            self.phase: 1})

    def predict(self, inputs, add_noise=True, training=1):
        """ Runs feedforward step on network to predict action. """
        out = self.session.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.phase: training})

        return out + add_noise*self.OU_noise()

    def predict_target(self, inputs):
        """ Runs feedforward step on target network to predict action. """
        return self.session.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
            self.phase: 1})

    def update_target_network(self):
        """ Updates target network parameters using Polyak averaging. """
        self.session.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        """ Returns number of trainable variables associated with actor network.
        """
        return self.num_trainable_vars

class CriticNetwork(object):
    """ Defines the critic network (Q-network) used in the DDPG algorithm. """

    def __init__(self, session, state_shape, action_shape, learning_rate, tau,
                 gamma, num_actor_vars):
        """ Initialize critic and target networks and update methods. """
        self.session = session
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Initialize critic network
        self.phase = tf.placeholder(tf.bool, name='phase_crt')
        self.inputs, self.action, self.out =  self.create_critic_network(
            self.phase)
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Initialize target critic network
        self.target_inputs, self.target_action, self.target_out = \
            self.create_critic_network(self.phase, prefix='tar_')
        self.target_network_params = tf.trainable_variables()[
            (len(self.network_params) + num_actor_vars):]

        # Define target update op
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) +
            tf.multiply(self.target_network_params[i], 1.0 - self.tau))
            for i in range(len(self.target_network_params))]

        # Define loss and optimization ops
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1]) # y_i
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out) # lol 
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            minimize(self.loss)

        # Define op for getting gradient of outputs wrt actions
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, phase, prefix=''):
        """ Constructs the critic network. """
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        N_OUT = 1
        OUT_INIT_MAG = 3e-4
        L2_SCALE = 1e-2

        # Build network using helper functions defined above
        state_in, to_fc1 = input_layer(self.state_shape, phase, # Jon note 11/18: No batch norm here?
            scope=prefix+'s_crt')
        action_in = tf.placeholder(shape=[None, self.action_shape],
            dtype=tf.float32, name='act_in')
        fc1 = dense_relu(state_in, N_HIDDEN_1, self.state_shape, phase,
            l2_scale=L2_SCALE, scope=prefix+'fc1_crt')
        fc2 = dense_relu_2input(fc1, action_in, N_HIDDEN_2, self.state_shape,
            l2_scale=L2_SCALE, scope=prefix+'fc2_crt')
        out = output_layer(fc2, N_OUT, OUT_INIT_MAG, act=None,
            scope=prefix+'out_crt')

        return state_in, action_in, out

    def train(self, inputs, action, predicted_q_value):
        """ Runs training step on critic network and returns feedforward output.
        """
        return self.session.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.phase: 1})

    def predict(self, inputs, action, training=1):
        """ Runs feedforward step on network to predict Q-value. """
        return self.session.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: training})

    def predict_target(self, inputs, action):
        """ Runs feedforward step on target network to predict Q-value. """
        return self.session.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.phase: 1})

    def action_gradients(self, inputs, actions):
        """ Returns gradient of outputs wrt actions. """
        return self.session.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.phase: 1})

    def update_target_network(self):
        """ Updates target network parameters using Polyak averaging. """
        self.session.run(self.update_target_network_params)
