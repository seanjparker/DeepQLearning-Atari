import tensorflow as tf
import numpy as np
import json
import math
from copy import deepcopy

from tensorflow.keras.models import Model

from agents.atari import construct_env, make_atari

from flask import Flask, render_template
from flask_socketio import SocketIO

from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey!'
app.config['DEBUG'] = True

socketio = SocketIO(app, logger=False)
tf_model = None


class ModelEnv:
    def __init__(self, _env, _model):
        self.env = _env
        self.model: Model = _model
        self.layer_names = [layer.name for layer in self.model.layers]
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
        new_obs, _, done, _ = env.step(action.numpy())
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        self.frames.append(new_obs)
        if done:
            new_obs = env.reset()
            new_obs = np.expand_dims(np.array(new_obs), axis=0)

        # # Shape (210, 160, 3)
        # org_obs = env.unwrapped.ale.getScreenRGB2()
        return construct_json(activations[3:6], activations[9:10], self.layer_names[3:6], new_obs[0, :, :, 0])


to_send_template = {
    'src': {
        'width': 84,
        'height': 84,
        'data': None
    },
    'layers': []
}
layer_template = {
    'name': None,
    'type': 'img',
    'col': 8,
    'row': 4,
    'output_shape': 20,
    'layer_data': []
}


def construct_json(activations, output_pred, layer_names, new_obs):
    to_send = deepcopy(to_send_template)
    # to_send['src']['data'] = org_obs.ravel().tolist()
    to_send['src']['data'] = new_obs.ravel().tolist()

    for layer_index in range(len(activations)):
        temp = deepcopy(layer_template)
        temp['name'] = layer_names[layer_index]
        number_of_filters = activations[layer_index].shape[-1]
        # Adds each filter of the current layer, converts from [0, 1] to [0, 255] and converts to a list
        for i in range(number_of_filters):
            temp['layer_data'].append((activations[layer_index][..., i] * 255).astype(int).ravel().tolist())
        to_send['layers'].append(temp)

        # Get the filter shape (assuming it is square)
        temp['output_shape'] = int(math.sqrt(len(temp['layer_data'][0])))
        temp['row'] = int(number_of_filters / temp['col'])

    q_values = deepcopy(layer_template)
    q_values['type'] = 'chart'
    q_values['layer_data'] = output_pred[0].tolist()[0]
    q_values['labels'] = tf_model.env.unwrapped.get_action_meanings()
    to_send['layers'].append(q_values)
    return json.dumps(to_send)


def create_env():
    created_env = make_atari('BreakoutNoFrameskip-v4')
    final_env = construct_env(created_env)
    return final_env


def load_tf_model_and_env():
    loaded_model = tf.keras.models.load_model('saved_model/dqn_breakout')
    loaded_model.summary()
    return loaded_model, create_env()


def create_activation_model(loaded_model):
    layer_outputs = [layer.output for layer in loaded_model.layers]
    return Model(inputs=loaded_model.input, outputs=layer_outputs)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('step')
def step_env():
    step_data = tf_model.step()
    socketio.emit('update', step_data)


@socketio.on('connect')
def connect():
    print('sending model layers back')
    socketio.emit('layers_update', tf_model.layer_names[3:6])


@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    model, env = load_tf_model_and_env()
    activation_model = create_activation_model(model)
    tf_model = ModelEnv(env, activation_model)
    socketio.run(app)
