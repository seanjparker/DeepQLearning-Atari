import tensorflow as tf
import numpy as np
from utils.atari import make_atari, construct_env


def test(env_name='PongNoFrameskip-v4', load_path=None):
    model = None
    if load_path is not None:
        model = tf.saved_model.load(load_path)

    print('Running trained model')
    env = make_atari(env_name)
    env = construct_env(env, frame_stack=True)
    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_reward = np.zeros(1)
    while True:
        actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_reward += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode reward={}'.format(episode_reward[i]))
                episode_reward[i] = 0


if __name__ == '__main__':
    test('./saved_model/dqn_pong')
