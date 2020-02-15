import tensorflow as tf
import numpy as np
import json
from copy import deepcopy

from tensorflow.keras.models import Model

from agents.atari import construct_env, make_atari

from flask import Flask, render_template
from flask_socketio import SocketIO

from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey!'
app.config['DEBUG'] = True

socketio = SocketIO(app, logger=True)
tf_model = None


class ModelEnv:
    def __init__(self, _env, _model):
        self.env = _env
        self.model = _model
        self.frames = deque([], maxlen=4)
        obs = self.env.reset()
        obs = np.expand_dims(np.array(obs), axis=0)
        self.frames.append(obs)
        self.frames.append(obs)
        self.frames.append(obs)
        self.frames.append(obs)

    def step(self):
        activations = activation_model.predict(np.concatenate(self.frames, axis=-1))
        action = tf.argmax(activations[9][0])
        new_obs, _, _, _ = env.step(action.numpy())
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        self.frames.append(new_obs)

        # activation = (activations[3][0] * 255).astype(int)  # 3 is the index of the first conv layer
        # y = activation[:, :, 0].ravel().tolist()  # y now contains the [0, 255] grayscale image, shape of output layer
        # return json.dumps(new_obs[0, :, :, 0].ravel().tolist())

        # print(np.shape(activations[4][0]))
        # print(np.shape(activations[5][0]))
        return construct_json(activations[3:6], new_obs[0, :, :, 0])


to_send_template = {
    'src': {
        'width': 84,
        'height': 84,
        'data': None
    },
    'layers': []
}
layer_template = {
    'col': 8,
    'row': 4,
    'output_shape': 20,
    'data': None
}


def construct_json(activations, new_obs):
    # print(np.shape(activations[1]))
    # print(np.shape(activations[2]))
    # print(np.shape(activations[3]))
    # print([np.shape(acti[0]) for acti in activations])

    to_send = deepcopy(to_send_template)
    to_send['src']['data'] = new_obs.ravel().tolist()
    # print(activations[1].shape[-1]) prints 64
    for i in range(activations[0].shape[-1]):
        # a = (activation_map[0] * 255).astype(int)
        # print(a[:, :, 0])
        # print(np.shape(activations[0][..., i]))
        # for filter_map in activation_map:
        #     print(np.shape(filter_map))
        # print(np.shape((activations[0][..., i][0] * 255).astype(int)))
        temp = deepcopy(layer_template)
        temp['data'] = (activations[0][..., i][0] * 255).astype(int)[:, :].ravel().tolist()
        to_send['layers'].append(temp)

    return json.dumps(to_send)


def load_tf_model_and_env():
    loaded_model = tf.keras.models.load_model('saved_model/dqn_breakout')
    loaded_model.summary()
    created_env = make_atari('BreakoutNoFrameskip-v4')
    final_env = construct_env(created_env)
    return loaded_model, final_env


def create_activation_model(loaded_model):
    layer_outputs = [layer.output for layer in loaded_model.layers]
    return Model(inputs=loaded_model.input, outputs=layer_outputs)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('step')
def step_env():
    print("stepping env")
    y = tf_model.step()
    socketio.emit('update', y)


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    model, env = load_tf_model_and_env()
    activation_model = create_activation_model(model)
    tf_model = ModelEnv(env, activation_model)
    socketio.run(app)
