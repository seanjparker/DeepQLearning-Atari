import tensorflow as tf
from dqn_train import learn
from atari import make_atari, construct_env


def main():
    env = make_atari('PongNoFrameskip-v4')
    env = construct_env(env, frame_stack=True)

    model = learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        learning_rate=1e-4,
        total_timesteps=int(1e7),
        buffer_size=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        checkpoint_path='./checkpoints/'
    )

    model.save('./saved_model/dqn_pong/')
    env.close()


if __name__ == '__main__':
    main()
