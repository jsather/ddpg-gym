""" ddpg.py is where the entire DDPG training process is run.

    Author: Jonathon Sather
    Last updated: 2/20/2018

    Implementation inspired by:
    https://github.com/pemami4911/deep-rl/tree/master/ddpg
    https://github.com/cbfinn/gps

    TODO: 1. Start making real thing! - agent_ros, plant_ros, E2C, pose_predictor
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

from networks import ActorNetwork, CriticNetwork
from replay_buffer import EpisodeReplayBuffer
import fake

import pdb

# from agent_ros import AgentROS# Make this. Interfaces arm with ROS
# from plant_ros import PlantROS # Make this. Interfaces with ROS plant environment.
#
# from E2C import e2c # Make this. Used for latent space rep.
# from pose_predictor import predictor1 # Make this. Used for predicting strawb pose.

def build_summaries():
    """ Sets up summary operations for use with Tensorboard. """

    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def learn(session,
          actor_network,
          critic_network,
          predictor_network,
          agent,
          plant,
          expert_demos=[],
          latent=False,
          latent_network=None,
          buffer_size=1000000,
          batch_size=64,
          trace_length=10,
          update_freq=5,
          pretrain_steps=10000,
          update_steps=1,
          max_episodes=50000,
          max_ep_steps=1000,
          summary_dir='./results/tf_ddpg'):
    """ Run the DDPG algorithm using networks passed as input and specified
        hyperparameters.
    """
    # Set up summary ops
    _, dirs, _ = os.walk(summary_dir).next()
    summary_dir = summary_dir + '/run' + str(len(dirs))

    summary_ops, summary_vars = build_summaries()

    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(summary_dir, session.graph)

    # Initialize target network weights
    actor_network.update_target_network()
    critic_network.update_target_network()

    # Initialize experience replay
    replay_memory = EpisodeReplayBuffer(buffer_size, init_data=expert_demos)

    render=False
    total_steps = 0

    for ep in range(max_episodes):
        print('Running episode...')

        # Set up episode
        # TODO: Make these methods!
        plant.new()
        agent.reset()
        o, j = agent.get_obs() # Returns camera image and joint angles

        ep_reward = 0
        ep_ave_max_q = 0

        if latent: # Convert camera image to latent space
           # TODO: Make this method! (and module lol)
            s = latent_network.convert(o)
        else:
            s = o[:]

        s = np.hstack((s, j))
        actor_hidden = 2 * (np.zeros([1, actor_network.hidden_1_size]),)
        episode_buffer = []

        for step in range(max_ep_steps):
            # Run actor network forward
            a, actor_hidden1 = actor_network.predict(
                np.reshape(s, (1, actor_network.state_shape)), actor_hidden,
                trace_length=1, batch_size=1, training=0)

            # TODO: Make this method! Consider making this into multiple methods.
            o2, j2 = agent.step(a, render=render)

            if latent: # Convert camera image to latent space
               # TODO: Make this method! (and module lol)
                s2 = latent_network.convert(o2)
            else:
                s2 = o2[:]

            s2 = np.hstack((s2.reshape((3,)), j2))

            # Get prediction confidence and corresponding reward
            # TODO: Make these methods!
            confidence, t = predictor_network.predict(o2)
            r = predictor_network.get_reward(confidence, t, a)

            # Add to trajectory
            episode_buffer.append(np.reshape(np.array([s,a,r,t,s2]), (1,-1)))
            total_steps += 1

            s = s2
            ep_reward += r
            actor_hidden = actor_hidden1

            if t:
                # Clear episode message from console
                sys.stdout.write('\033[F\033[K')

                render=False

                # TODO: Periodically save model

                # Store episode in replay buffer
                episode_buffer_array = np.array(episode_buffer)
                episode_buffer = list(zip(episode_buffer_array))
                replay_memory.add(episode_buffer)

                # Sample from buffer and train when appropriate
                if total_steps > pretrain_steps and ep % update_freq == 0:
                    print('Training network...')
                    render=True

                    for update in range(update_steps):
                        train_hidden = 2 * (np.zeros([batch_size,
                            actor_network.hidden_1_size]),)

                        # Sample trajectories from replay buffer
                        s_batch, a_batch, r_batch, t_batch, s2_batch = \
                            replay_memory.sample_batch(batch_size, trace_length)

                        # Calculate target values
                        target_q = critic_network.predict_target(s2_batch,
                            actor_network.predict_target(s2_batch, train_hidden,
                                trace_length=trace_length,
                                batch_size=batch_size),
                            train_hidden, trace_length=trace_length,
                            batch_size=batch_size)

                        # Calculate training values
                        end_mask = -(t_batch - 1)
                        y_i = r_batch + critic_network.gamma*target_q*end_mask

                        # Update critic given targets
                        predicted_q_value, _ = critic_network.train(s_batch,
                            a_batch, y_i, train_hidden,
                            trace_length=trace_length, batch_size=batch_size)
                        ep_ave_max_q += np.amax(predicted_q_value)

                        # Update actor policy using sampled gradient
                        a_outs, _ = actor_network.predict(s_batch, train_hidden,
                            trace_length=trace_length, batch_size=batch_size,
                            add_noise=False)
                        grads = critic_network.action_gradients(s_batch, a_outs,
                            train_hidden, trace_length=trace_length,
                            batch_size=batch_size)
                        actor_network.train(s_batch, grads[0], train_hidden,
                            trace_length=trace_length, batch_size=batch_size)

                        # Update target networks
                        actor_network.update_target_network()
                        critic_network.update_target_network()

                    # Clear training message from console
                    sys.stdout.write('\033[F\033[K')

                # Log data and start new episode
                summary_str = session.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(step)})

                writer.add_summary(summary_str, ep)
                writer.flush()

                print('| Reward: {:d} | Episode {:d} | Qmax: {:4f}'.format(
                    int(ep_reward), ep, (ep_ave_max_q / float(step))))

                break


def main(args):

    with tf.Session() as session:
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))

        # initialize ROS interface
        agent = fake.fake_agent()
        plant = fake.fake_plant()

        state_shape = agent.get_state_shape()
        action_shape = agent.get_action_shape()
        action_bound = agent.get_action_bound()

        # initialize function approximators
        actor_network = ActorNetwork(session, state_shape, action_shape,
            action_bound, float(args['actor_lr']), float(args['tau']),
            loss_mask=True)
        critic_network = CriticNetwork(session, state_shape, action_shape,
            float(args['critic_lr']), float(args['tau']), float(args['gamma']),
            actor_network.get_num_trainable_vars(), loss_mask=True)
        predictor_network = fake.fake_predictor()
        latent_network = fake.fake_latent()

        learn(session,
              actor_network,
              critic_network,
              predictor_network,
              agent,
              plant,
              latent_network=latent_network,
              buffer_size=int(args['buffer_size']),
              batch_size=int(args['batch_size']),
              trace_length=int(args['trace_length']),
              update_freq=int(args['update_freq']),
              pretrain_steps=int(args['pretrain_steps']),
              update_steps=int(args['update_steps']),
              max_episodes=int(args['max_episodes']),
              max_ep_steps=int(args['max_episode_len']),
              summary_dir=args['summary_dir'])

if __name__ == '__main__':
    # Parse command-line argumnets and run algorithm
    parser = ArgumentParser(description='provide arguments for DDPG algorithm')

    # Learning parameters
    parser.add_argument('--actor-lr', help='actor network learning rate',
        default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate',
        default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates',
        default=0.99)
    parser.add_argument('--tau', help='soft target update parameter',
        default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer',
        default=1000)
    parser.add_argument('--batch-size',
        help='size of minibatch for minibatch-SGD', default=4)
    parser.add_argument('--trace-length',
        help='length of traces for training rnn', default=8)
    parser.add_argument('--update-freq',
        help='number of episodes between each training update', default=1)
    parser.add_argument('--pretrain-steps',
        help='number of steps in environment before start training prodecure',
        default=10000)
    parser.add_argument('--update-steps',
        help='number of gradient descent steps per update', default=40)

    # Experiment parameters
    parser.add_argument('--random-seed', help='random seed for repeatability',
        default=0)
    parser.add_argument('--max-episodes',
        help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode',
        default=1000)
    parser.add_argument('--summary-dir',
        help='directory for storing tensorboard info',
        default='../results/ddpg_gym')

    args = vars(parser.parse_args())

    main(args)
