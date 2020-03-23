import tensorflow as tf


class LinearSchedule:
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def step_to(self, t):
        return tf.constant(self.initial_p + min(float(t) / self.schedule_timesteps, 1.0) * (self.final_p - self.initial_p))
