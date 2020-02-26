import tensorflow as tf


def conv(convs=None, **conv_kwargs):
    """
    convolutional layer
    Parameters:
    ----------
    conv: list of triples (filter_number, filter_size, stride)
    Returns:
        function input is tf tensor, returns output of conv layer
    """

    if convs is None:
        convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

    def network_fn(input_shape):
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = tf.cast(x_input, tf.float32) / 255.0
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                h = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn


def build_q_func(network, hiddens=None, **network_kwargs):
    if hiddens is None:
        hiddens = [256]
    if isinstance(network, str):
        network = conv(network)(**network_kwargs)

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
