import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, next_obs, done):
        experience = (obs, action, reward, next_obs, done)
        self.buffer.append(experience)

    def encode_samples(self, indexes):
        obses, actions, rewards, next_obses, dones = [], [], [], [], []
        data = self.buffer[0]
        obs_dtype = data[0].dtype
        action_dtype = data[1].dtype

        for i in indexes:
            data = self.buffer[i]
            obs, action, reward, next_obs, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_obses.append(np.array(next_obs, copy=False))
            dones.append(done)

        return np.array(obses, dtype=obs_dtype), \
            np.array(actions, dtype=action_dtype), \
            np.array(rewards, dtype=np.float32), \
            np.array(next_obses, dtype=obs_dtype), \
            np.array(dones, dtype=np.float32)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexs = np.random.choice(np.arange(buffer_size), size=batch_size)

        return self.encode_samples(indexs)
