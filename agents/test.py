import tensorflow as tf
import numpy as np
from atari import make_atari, construct_env


def test(load_path=None):
    model = None
    if load_path is not None:
        model = tf.saved_model.load(load_path)

    print("Running trained model")
    env = make_atari('PongNoFrameskip-v4')
    env = construct_env(env, frame_stack=True)
    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = np.zeros(1)
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0


if __name__ == '__main__':
    test('./saved_model/dqn_pong')
