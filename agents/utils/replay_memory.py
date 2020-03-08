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

    def get_samples_from_indexes(self, indexes) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs0, actions, rewards, obs1, dones = [], [], [], [], []
        data = self.buffer[0]
        obs_dtype = data[0].dtype
        action_dtype = data[1].dtype

        for i in indexes:
            data = self.buffer[i]
            obs_t, action, reward, obs_t1, done = data

            # copy = False, to prevent creating copies of all the objects in the memory!
            obs0.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obs1.append(np.array(obs_t1, copy=False))
            dones.append(done)

        return np.array(obs0, dtype=obs_dtype), \
            np.array(actions, dtype=action_dtype), \
            np.array(rewards, dtype=np.float32), \
            np.array(obs1, dtype=obs_dtype), \
            np.array(dones, dtype=np.float32)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexs = np.random.choice(np.arange(buffer_size), size=batch_size)

        return self.get_samples_from_indexes(indexs)
