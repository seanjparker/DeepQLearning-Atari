import tensorflow as tf
import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, next_obs, done):
        # Store the sample (s, a, r, s', d) sample in replay memory
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample_from_indexes(self, indexes) -> [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        obs0, actions, rewards, obs1, dones = [], [], [], [], []
        obs_dtype = self.buffer[0][0].dtype
        action_dtype = self.buffer[0][1].dtype

        for i in indexes:
            data = self.buffer[i]
            obs_t, action, reward, obs_t1, done = data

            # copy = False, to prevent creating copies of all the objects in the memory!
            obs0.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obs1.append(np.array(obs_t1, copy=False))
            dones.append(done)

        return tf.constant(np.array(obs0, dtype=obs_dtype)), \
            tf.constant(np.array(actions, dtype=action_dtype)), \
            tf.constant(np.array(rewards, dtype=np.float32)), \
            tf.constant(np.array(obs1, dtype=obs_dtype)), \
            tf.constant(np.array(dones, dtype=np.float32))

    def sample(self, batch_size):
        chosen_indexes = np.random.permutation(np.arange(len(self.buffer)))[:batch_size]

        return self.sample_from_indexes(chosen_indexes)
