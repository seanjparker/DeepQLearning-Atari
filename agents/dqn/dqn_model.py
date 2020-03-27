from typing import Callable, Any
import tensorflow as tf


@tf.function()
def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


class DeepQ(tf.Module):
    """
    Class that handles training a DQN
    """
    def __init__(self, model_builder: Callable[[Any, Any, Any], tf.keras.Model], observation_shape, num_actions,
                 learning_rate, gamma=1.0, double_dqn=False):
        super().__init__()
        self.num_actions = num_actions
        self.gamma = gamma

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.loss = huber_loss
        self.train_loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        self.q_network = model_builder(observation_shape, num_actions, 'q_network')
        self.target_q_network = model_builder(observation_shape, num_actions, 'target_q_network')

        self.q_network.compile(optimizer=self.optimizer, loss=self.loss)
        self.eps = tf.Variable(0.0, name="eps")

        self.double_dqn = double_dqn

    @tf.function()
    def step(self, observation, stochastic=True, update_eps=-1):
        """Function to chose an action given an observation

        Parameters
        ----------
        observation: tensor
            Observation from the environment
        stochastic: bool
            when false, all actions are deterministic (default True)
        update_eps: float
            update epsilon a new value, if negative not update happens (default is not update)

        Returns
        -------
            Tensor of data type tf.int64 and shape batch_size with an action to be performed for
            every element of the batch.
        """
        q_values = self.q_network(observation)
        deterministic_actions = tf.argmax(q_values, axis=1)
        batch_size = tf.shape(observation)[0]

        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
        choose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
        stochastic_actions = tf.where(choose_random, random_actions, deterministic_actions)

        output_actions = stochastic_actions if stochastic else deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)

        return output_actions, None, None, None

    @tf.function()
    def get_best_q_action(self, obs1, q_tp1_from_target):
        """ Gets the best q-value, we use a different equation based on if we are using a normal dqn or double dqn
        """
        if self.double_dqn:
            q_tp1_best = tf.argmax(self.q_network(obs1), 1)
            return tf.reduce_sum(q_tp1_from_target * tf.one_hot(
                q_tp1_best, self.num_actions, dtype=tf.float32), 1)
        else:
            return tf.reduce_max(q_tp1_from_target, 1)

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, weights):
        """Function that takes a transition (s, a, r, s') and optimises for the Bellman equation

        where d is a boolean that represents if the episode is done

        td_error = Q(s,a) - (r + gamma * (1 - d) * max_a' Q(s', a'))

        loss = huber_loss(td_error)

        Parameters
        ----------
        obs0: object
            a batch of observations at timestep t
        actions: np.array
            actions that were selected upon seeing obs_t.
            dtype must be int32 and shape must be (batch_size,)
        rewards: np.array
            immediate reward attained after executing those actions
            dtype must be float32 and shape must be (batch_size,)
        obs1: object
            observations at timestep, t + 1
        dones: np.array
            1 if obs0 was the last observation in the episode and 0 otherwise
            obs1 gets ignored, dtype must be float32 and shape must be (batch_size,)
        weights: np.array
            weights for every element of the batch (gradient is multiplied
            by the weight) dtype is float32 and shape is (batch_size,)

        Returns
        -------
        td_error: np.array, dtype is float32 and shape is (batch_size,)
            a list of differences between Q(s,a) and the target in Bellman's equation.
        """

        # Use graident tape to record the gradients for the update
        with tf.GradientTape() as tape:

            # Perform a forward pass through the network using the current observation
            q_t = self.q_network(obs0)

            # Perform one-hot encoding of the q-value result based on selected action
            q_t_weighted = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), 1)

            # Perform forward pass through target network
            q_tp_1 = self.target_q_network(obs1)

            # Get the best q-value from the target network pass
            # If we are using a double dqn then use find q_best based on target and q-network predictions
            q_tp_1_best = self.get_best_q_action(obs1, q_tp_1)
            dones = 1.0 - tf.cast(dones, q_tp_1_best.dtype)

            q_tp_1_best_after_done_mask = dones * q_tp_1_best

            # Calculate the losses from the bellman equation
            q_t_selected_target_value = rewards + self.gamma * q_tp_1_best_after_done_mask
            td_loss = q_t_weighted - tf.stop_gradient(q_t_selected_target_value)

            # Clip the losses using the huber loss function
            # See https://en.wikipedia.org/wiki/Huber_loss
            losses = huber_loss(td_loss)
            weighted_loss = tf.reduce_mean(weights * losses)

        # Compute the gradients from the gradient tape
        grads = tape.gradient(weighted_loss, self.q_network.trainable_variables)
        grads_and_vars = zip(grads, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        # Log the training loss for the current step using tensorboard
        self.train_loss_metrics(td_loss)
        return td_loss

    @tf.function(autograph=False)
    def update_target(self):
        """Copies the params from the q function to target func

        Q-learning optimises the error:

            Q(s,a) - (r + gamma * max_a' * Q'(s', a'))

        Where Q' is lagging behind Q to stablize the learning
        """
        q_vars = self.q_network.trainable_variables
        target_q_vars = self.target_q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)
