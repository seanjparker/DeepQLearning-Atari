import os.path as osp

import tensorflow as tf
import numpy as np

from schedule import LinearSchedule
from replay_memory import ReplayMemory
from dqn_model import DeepQ
from dqn_model_builder import build_q_func


def learn(env,
          network,
          learning_rate=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          **network_kwargs) -> tf.keras.Model:
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models
        in agents.dqn_model_builder (conv_only). If a function, should take an observation tensor and
        return a latent variable tensor, which will be mapped to the Q function heads
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
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    checkpoint_path: str
        the os path where to store the checkpoints on the system
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    model: an instance tf.Module that contains the trained model
    """
    # Create all the functions necessary to train the model
    q_func = build_q_func(network, **network_kwargs)

    model = DeepQ(
        model_builder=q_func,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        learning_rate=learning_rate,
        grad_norm_clipping=10,
        gamma=gamma
    )

    manager = None
    if checkpoint_path is not None:
        load_path = osp.expanduser(checkpoint_path)
        ckpt = tf.train.Checkpoint(model=model.q_network)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))

    # Create the replay buffer
    replay_buffer = ReplayMemory(buffer_size)
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    model.update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()

    obs = np.expand_dims(np.array(obs), axis=0)
    reset = True

    for t in range(total_timesteps):
        update_eps = tf.constant(exploration.value(t))
        update_param_noise_threshold = 0.
        action, _, _, _ = model.step(tf.constant(obs), update_eps=update_eps)
        action = action[0].numpy()
        reset = False
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        replay_buffer.add(obs[0], action, rew, new_obs[0], float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            obs = env.reset()
            obs = np.expand_dims(np.array(obs), axis=0)
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, batch_idxes = tf.ones_like(rewards), None

            obses_t, obses_tp1 = tf.constant(obses_t), tf.constant(obses_tp1)
            actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
            td_errors = model.train(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            model.update_target()

        mean_100ep_reward = np.round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            format_str = "steps: {}, episodes: {}, mean 100 ep reward: {}, %time spent expl: {}"
            print(format_str.format(t, num_episodes, mean_100ep_reward, int(100 * exploration.value(t))))

        if done and checkpoint_path is not None and t % checkpoint_freq == 0:
            manager.save()

    return model.q_network
