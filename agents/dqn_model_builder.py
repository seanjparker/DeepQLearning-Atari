import tensorflow as tf

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


@register("conv_only")
def conv_only(convs=None, **conv_kwargs):
    """
    convolutions-only net
    Parameters:
    ----------
    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    Returns:
    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    """

    if convs is None:
        convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = tf.cast(x_input, tf.float32) / 255.
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                h = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network
    return network_fn


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))


def build_q_func(network, hiddens=None, layer_norm=False, **network_kwargs):
    if hiddens is None:
        hiddens = [256]
    if isinstance(network, str):
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_shape, num_actions):
        # the sub Functional model which does not include the top layer.
        model = network(input_shape)

        # wrapping the sub Functional model with layers that compute action scores into another Functional model.
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]

        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            action_out = latent
            for hidden in hiddens:
                action_out = tf.keras.layers.Dense(units=hidden, activation=None)(action_out)
                if layer_norm:
                    action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)

        q_out = action_scores
        model = tf.keras.Model(inputs=model.inputs, outputs=[q_out])
        model.summary()
        return model

    return q_func_builder
