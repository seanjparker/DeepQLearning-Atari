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

    def network_builder(input_shape, input_dtype=tf.uint8):
        # Ensure the network input is taken as uint8, since that is the datatype we
        # store the experience as in the experience replay memory

        # Otherwise, we can pass the input data type as ann argument
        network_input = tf.keras.Input(shape=input_shape, dtype=input_dtype)

        # Normalise the network input to the range [0, 1] from [0, 255]
        layer = tf.cast(network_input, tf.float32) / 255.0
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                layer = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(layer)

        network = tf.keras.Model(inputs=[network_input], outputs=[layer])
        return network
    return network_builder
