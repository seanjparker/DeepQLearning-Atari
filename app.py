import tensorflow as tf
import numpy as np
import json
import math
from copy import deepcopy

from tensorflow.keras.models import Model

from agents.utils.atari import construct_env

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
        self.frames.extend([obs] * 4)

        self.avaliable_models = ['Pong', 'Breakout']

    def step(self):
        activations = self.model.predict(np.concatenate(self.frames, axis=-1))
        # print(activations[8])
        action = tf.argmax(activations[9][0])
        # print(action)
        new_obs, _, done, _ = self.env.step(action.numpy())
        new_obs = np.expand_dims(np.array(new_obs), axis=0)
        self.frames.append(new_obs)
        if done:
            new_obs = self.env.reset()
            new_obs = np.expand_dims(np.array(new_obs), axis=0)

        # # Shape (210, 160, 3)
        # org_obs = env.unwrapped.ale.getScreenRGB2()
        return construct_json(activations[3:6], activations[9], self.layer_names[3:6], new_obs[0, :, :, 0])

    def reset(self, new_env, new_model):
        self.env = new_env
        self.model = new_model
        self.frames = deque([], maxlen=4)

        obs = self.env.reset()
        obs = np.expand_dims(np.array(obs), axis=0)
        self.frames.extend([obs] * 4)


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
    q_values['layer_data'] = output_pred[0].tolist()
    q_values['labels'] = tf_model.env.unwrapped.get_action_meanings()
    to_send['layers'].append(q_values)
    return json.dumps(to_send)


def create_env(game_name):
    created_env = construct_env(game_name + 'NoFrameskip-v4', frame_skip=4)
    return created_env


def load_tf_model_and_env(game_name):
    # Ensure the game name is lowercase for loading the model
    game_name_model_load = game_name.lower()
    loaded_model = tf.keras.models.load_model('saved_model/dqn_' + game_name_model_load)
    loaded_model.summary()
    print(game_name_model_load)
    return loaded_model, create_env(game_name)


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


@socketio.on('switchModel')
def switch_model(data):
    new_model_index = int(data['new_model'])
    new_model_name = tf_model.avaliable_models[new_model_index]
    print(new_model_name)
    new_model, new_env = load_tf_model_and_env(new_model_name)
    new_activation_model = create_activation_model(new_model)

    tf_model.reset(new_env, new_activation_model)
    step_env()


@socketio.on('connect')
def connect():
    socketio.emit('layers_update', {
        'layer_names': tf_model.layer_names[3:6],
        'model_names': tf_model.avaliable_models
    })


@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    model, env = load_tf_model_and_env('Pong')
    activation_model = create_activation_model(model)
    tf_model = ModelEnv(env, activation_model)
    socketio.run(app)
