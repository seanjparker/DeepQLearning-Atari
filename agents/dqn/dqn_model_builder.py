import tensorflow as tf
from utils.network_builder import conv


def build_q_func(network, hiddens=None):
    if hiddens is None:
        hiddens = [256]
    network = conv(convs=network)

    def q_func_builder_func(input_shape, num_actions) -> tf.keras.Model:
        # the model (built using the functional API) that is just for the Q-Network function approximator
        model = network(input_shape)

        initial = model.outputs[0]
        initial = tf.keras.layers.Flatten()(initial)

        with tf.name_scope("action_value"):
            action_out = initial
            for hidden in hiddens:
                action_out = tf.keras.layers.Dense(units=hidden, activation='relu')(action_out)
            action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)

        q_out = action_scores
        model = tf.keras.Model(inputs=model.inputs, outputs=[q_out])
        model.summary()
        return model

    return q_func_builder_func
