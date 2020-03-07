import tensorflow as tf
from utils.network_builder import conv


def build_dueling_q_func(network, hiddens=None):
    if hiddens is None:
        hiddens = [256]
    network = conv(convs=network)

    def dueling_q_builder_func(input_shape, num_actions, name) -> tf.keras.Model:
        # the model (built using the functional API) that is just for the Q-Network function approximator
        with tf.name_scope(name):
            model = network(input_shape)

            initial = model.outputs[0]
            initial = tf.keras.layers.Flatten()(initial)

            # When using a dueling q network, the estimation of Q(s, a) is split into two parts in the network:
            # Defined by Q(s, a) = A(s, a) + V(s)
            # A(s, a) estimates the advantage of taking each action (given the state, s) A(s, a)
            # V(s) estimates the state-value
            # These two paths are combined back into a single stream (see aggregation scoped op below)

            with tf.name_scope("advantage_fc"):
                advantage_layer = initial
                for hidden in hiddens:
                    advantage_layer = tf.keras.layers.Dense(units=hidden, activation='relu')(advantage_layer)
                advantage_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(advantage_layer)

            with tf.name_scope("valuc_fc"):
                value_layer = initial
                for hidden in hiddens:
                    value_layer = tf.keras.layers.Dense(units=hidden, activation='relu')(value_layer)
                value_scores = tf.keras.layers.Dense(units=1, activation=None)(value_layer)

            with tf.name_scope("aggregation"):
                q_out = value_scores + tf.subtract(
                    advantage_scores, tf.reduce_mean(advantage_scores, axis=1, keepdims=True)
                )

        model = tf.keras.Model(inputs=model.inputs, outputs=[q_out])
        model.summary()
        return model

    return dueling_q_builder_func
