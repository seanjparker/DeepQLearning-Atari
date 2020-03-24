import tensorflow as tf


class LinearSchedule:
    def __init__(self, total_timesteps, final_prob, initial_prob=1.0):
        self.total_timesteps = total_timesteps
        self.final_prob = final_prob
        self.initial_prob = initial_prob

    def step_to(self, c_t):
        frac = min(float(c_t) / self.total_timesteps, 1.0)
        annealed_linear_prob = self.initial_prob + frac * (self.final_prob - self.initial_prob)
        return tf.constant(annealed_linear_prob)
