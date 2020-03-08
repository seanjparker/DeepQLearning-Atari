from .dqn_train import train_model
from utils.atari import construct_env


def train():
    env = construct_env('BreakoutNoFrameskip-v4', frame_stack=True,
                        record_video=True, record_video_steps=500, frame_skip=4)

    model = train_model(
        env,
        conv_layers=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
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

    model.save('./saved_model/dqn_breakout/')
    env.close()
