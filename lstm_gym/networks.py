""" networks.py contains the classes for defining the actor and critic networks
    used in the DDPG algorithm.

    Author: Jonathon Sather
    Last updated: 2/26/2018

    Uses most parameters from original DDPG paper:
    'Continuous Control in Deep Reinforcement Learning' by Lillicrap, et al.
    (arXiv:1509.02971)

    Uses similar architecture to that used in RDPG paper:
    'Memory-based Control with Recurrent Neural Networks' by Hees, Hunt, et al.

    Uses LSTM update scheme from DRQN 'Doom' paper:
    'Playing FPS Games with Deep Reinforcement Learning' by Lample and Chaplot

    Modular structure influence by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg
"""

import tensorflow as tf
import numpy as np
import pdb

from noise import OrnsteinUhlenbeckActionNoise

#------------------------------ Helper functions ------------------------------#
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

def lstm_layer(x, size, batch_size, scope='layer'):
    """ Creates LSTM layer with ____ ____ ___.
    """
    with tf.variable_scope(scope):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=size,
            state_is_tuple=True)
        init_state = rnn_cell.zero_state(batch_size, tf.float32)
        rnn, rnn_state = tf.nn.dynamic_rnn(inputs=x, cell=rnn_cell,
            dtype=tf.float32, initial_state=init_state, scope='rnn')
        rnn = tf.reshape(rnn, shape=[-1, size])

        return rnn, rnn_state, init_state

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

#------------------------------ Network classes -------------------------------#
class ActorNetwork(object):
    """ Defines the actor network (Q-network optimizer) used in the DDPG
        algorithm.
    """
    def __init__(self, session, state_shape, action_shape, action_bound,
                 learning_rate, tau, loss_mask=True):
        """ Initialize actor and target networks and update methods. """
        self.session = session
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        self.hidden_1_size = 400
        self.hidden_2_size = 300

        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_act')
        self.trace_length = tf.placeholder(tf.int32, name='trace_act')
        self.phase = tf.placeholder(tf.bool, name='phase_act')

        # Initialize addititve Ornstein Uhlenbeck noise
        self.OU_noise = \
            OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_shape))

        # Initialize actor network
        self.inputs, self.out, self.scaled_out, self.lstm_state, \
            self.lstm_init_state = self.create_actor_network()
        self.network_params = tf.trainable_variables()

        # Initialize target actor network
        self.target_inputs, self.target_out, self.target_scaled_out, \
            self.target_lstm_state, self.target_lstm_init_state = \
                self.create_actor_network(prefix='tar_')
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

        if loss_mask:
            # Mask first half of losses for each trace per Lample & Charlot 2016
            self.maskA = tf.zeros([self.batch_size, self.trace_length//2])
            self.maskB = tf.ones([self.batch_size, self.trace_length//2])
            self.mask = tf.concat([self.maskA, self.maskB], 1)
            self.mask = tf.reshape(self.mask, [-1])
            self.action_gradient_adjusted = self.action_gradient * self.mask
        else:
            self.action_gradient_adjusted = self.action_gradient

        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params,
            -self.action_gradient_adjusted)
        self.actor_gradients = list(map(lambda x: tf.div(x,
            tf.cast(self.batch_size, tf.float32)),
            self.unnormalized_actor_gradients))

        # Define optimization op
        # TODO: Only update BN params when needed instead of all the time!
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = \
            len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, prefix=''):
        """ Constructs the actor network. Phase is boolean tensor corresponding
            to whether in training or testing phase.
        """
        OUT_INIT_MAG = 3e-3

        # Construct network using helper functions defined above
        # inputs, to_fc1 = input_layer(self.state_shape, self.phase,
        #     scope=prefix+'in_act')
        # Make new input helper with BN + reshape
        state_in = tf.placeholder(
            shape=[None, self.state_shape], dtype=tf.float32,
            name='state_in')
        state_reshape = tf.reshape(state_in, [self.batch_size,
            self.trace_length, self.state_shape])
        lstm1, lstm_state, lstm_init_state = lstm_layer(state_reshape,
            self.hidden_1_size, self.batch_size, scope=prefix+'lstm1_act')
        fc1 = dense_relu(lstm1, self.hidden_2_size, self.hidden_1_size,
            self.phase, scope=prefix+'fc1_act')
        out = output_layer(fc1, self.action_shape, OUT_INIT_MAG, act=tf.tanh,
            scope=prefix+'out_act')
        scaled_out = tf.multiply(out, self.action_bound)

        return state_in, out, scaled_out, lstm_state, lstm_init_state

    def train(self, inputs, action_gradient, init_state, trace_length=1,
        batch_size=1):
        """ Runs training step on actor network. """
        self.session.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: action_gradient,
            self.phase: 1,
            self.trace_length: trace_length,
            self.batch_size: batch_size,
            self.lstm_init_state: init_state})

    def predict(self, inputs, init_state, trace_length=1, batch_size=1,
        add_noise=True, training=1):
        """ Runs feedforward step on network to predict action. Also returns
            hidden state of LSTM.
        """
        out, hidden = self.session.run([self.scaled_out, self.lstm_state],
            feed_dict={
                self.inputs: inputs,
                self.phase: training,
                self.trace_length: trace_length,
                self.batch_size: batch_size,
                self.lstm_init_state: init_state})
        out = out + add_noise*self.OU_noise()

        return (out, hidden)

    def predict_target(self, inputs, init_state, trace_length=1, batch_size=1):
        """ Runs feedforward step on target network to predict action. """
        return self.session.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
            self.phase: 1,
            self.trace_length: trace_length,
            self.batch_size: batch_size,
            self.target_lstm_init_state: init_state})

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
                 gamma, num_actor_vars, loss_mask=True):
        """ Initialize critic and target networks and update methods. """
        self.session = session
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.hidden_1_size = 400
        self.hidden_2_size = 300

        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_crt')
        self.trace_length = tf.placeholder(tf.int32, name='trace_crt')
        self.phase = tf.placeholder(tf.bool, name='phase_crt')

        # Initialize critic network
        self.inputs, self.action, self.out, self.lstm_state, \
            self.lstm_init_state = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Initialize target critic network
        self.target_inputs, self.target_action, self.target_out, \
            self.target_lstm_state, self.target_lstm_init_state = \
                self.create_critic_network(prefix='tar_')
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
        self.td_error = tf.square(self.predicted_q_value - self.out)

        if loss_mask:
            # Mask first half of losses for each trace per Lample & Charlot 2016
            self.maskA = tf.zeros([self.batch_size, self.trace_length//2])
            self.maskB = tf.ones([self.batch_size, self.trace_length//2])
            self.mask = tf.concat([self.maskA, self.maskB], 1)
            self.mask = tf.reshape(self.mask, [-1])
            self.loss = tf.reduce_mean(self.td_error * self.mask)
        else:
            self.loss = tf.reduce_mean(self.td_error)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            minimize(self.loss)

        # Define op for getting gradient of outputs wrt actions
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, prefix=''):
        """ Constructs the critic network. """
        OUT_INIT_MAG = 3e-4
        L2_SCALE = 1e-2

        # Build network using helper functions defined above
        # TODO: Make helper function to accept input and reshape and BN
        state_in = tf.placeholder(
            shape=[None, self.state_shape], dtype=tf.float32,
            name='state_in')
        state_reshape = tf.reshape(state_in, [self.batch_size,
            self.trace_length, self.state_shape])
        action_in = tf.placeholder(shape=[None, self.action_shape],
            dtype=tf.float32, name='act_in')
        lstm1, lstm_state, lstm_init_state = lstm_layer(state_reshape,
            self.hidden_1_size, self.batch_size, scope=prefix+'lstm1_crt')
        fc1 = dense_relu_2input(lstm1, action_in, self.hidden_2_size,
            self.state_shape, l2_scale=L2_SCALE, scope=prefix+'fc1_crt')
        out = output_layer(fc1, 1, OUT_INIT_MAG, act=None,
            scope=prefix+'out_crt')

        return state_in, action_in, out, lstm_state, lstm_init_state

    def train(self, inputs, action, predicted_q_value, init_state,
        trace_length=1, batch_size=1):
        """ Runs training step on critic network and returns feedforward output.
        """
        return self.session.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.phase: 1,
            self.trace_length: trace_length,
            self.batch_size: batch_size,
            self.lstm_init_state: init_state})

    def predict(self, inputs, action, init_state, trace_length=1, batch_size=1,
        training=1):
        """ Runs feedforward step on network to predict Q-value. """
        return self.session.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: training,
            self.trace_length: trace_length,
            self.batch_size: batch_size,
            self.lstm_init_state: init_state})

    def predict_target(self, inputs, action, init_state, trace_length=1,
        batch_size=1):
        """ Runs feedforward step on target network to predict Q-value. """
        return self.session.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.phase: 1,
            self.trace_length: trace_length,
            self.batch_size: batch_size,
            self.target_lstm_init_state: init_state})

    def action_gradients(self, inputs, actions, init_state, trace_length=1,
        batch_size=1):
        """ Returns gradient of outputs wrt actions. """
        return self.session.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.phase: 1,
            self.trace_length: trace_length,
            self.batch_size: batch_size,
            self.lstm_init_state: init_state})

    def update_target_network(self):
        """ Updates target network parameters using Polyak averaging. """
        self.session.run(self.update_target_network_params)
