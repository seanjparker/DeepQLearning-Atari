import os.path as osp

import tensorflow as tf
import numpy as np

from utils.schedule import LinearSchedule
from utils.replay_memory import ReplayMemory
from .dqn_model import DeepQ
from .dqn_model_builder import build_q_func

import datetime


def train_model(env,
                conv_layers,
                learning_rate=5e-4,
                total_timesteps=100000,
                buffer_size=50000,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                train_freq=1,
                batch_size=32,
                print_freq=1,
                checkpoint_freq=100000,
                checkpoint_path=None,
                learning_starts=1000,
                gamma=1.0,
                target_network_update_freq=500,
                double_dqn=False,
                **network_kwargs) -> tf.keras.Model:
    """Train a DQN model.

    Parameters
    -------
    env: gym.Env
        openai gym
    conv_layers: list
        a list of triples that defines the conv network
    learning_rate: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to store a checkpoint during training
    checkpoint_path: str
        the fs path for storing the checkpoints
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    double_dqn: bool
        specifies if double q-learning is used during training
    Returns
    -------
    dqn: an instance of tf.Module that contains the trained model
    """
    q_func = build_q_func(conv_layers, **network_kwargs)

    dqn = DeepQ(
        model_builder=q_func,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        double_dqn=double_dqn
    )

    manager = None
    if checkpoint_path is not None:
        load_path = osp.expanduser(checkpoint_path)
        ckpt = tf.train.Checkpoint(model=dqn.q_network)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Create the replay buffer
    replay_buffer = ReplayMemory(buffer_size)
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    dqn.update_target()

    episode_rewards = [0.0]
    obs = env.reset()

    obs = np.expand_dims(np.array(obs), axis=0)
    reset = True

    for t in range(total_timesteps):
        update_eps = tf.constant(exploration.value(t))

        action, _, _, _ = dqn.step(tf.constant(obs), update_eps=update_eps)
        action = action[0].numpy()
        reset = False
        new_obs, reward, done, _ = env.step(action)
        # Store transition in the replay buffer.
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        replay_buffer.add(obs[0], action, reward, new_obs[0], float(done))
        obs = new_obs

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            obs = np.expand_dims(np.array(obs), axis=0)
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, _ = tf.ones_like(rewards), None

            obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
            actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
            td_loss = dqn.train(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network every target_network_update_freq steps
            dqn.update_target()

        num_episodes = len(episode_rewards) - 1
        if done and print_freq is not None and num_episodes % print_freq == 0:
            mean_100ep_reward = np.round(np.mean(episode_rewards[-102:-2]), 1)
            format_str = "steps: {}, episodes: {}, mean 100 ep reward: {}, episode reward: {}, %time spent expl: {}"
            print(format_str.format(t, num_episodes, mean_100ep_reward, episode_rewards[-2],
                                    int(100 * exploration.value(t))))

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', dqn.train_loss_metrics.result(), step=t)
                tf.summary.scalar('reward', episode_rewards[-2], step=t)

            # Every episode, reset the loss metric
            dqn.train_loss_metrics.reset_states()

        if checkpoint_path is not None and t % checkpoint_freq == 0:
            manager.save()

    return dqn.q_network
