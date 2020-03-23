from dueling_dqn.ddqn_train import train_model
from utils.atari import construct_env


def train(use_double_dqn=False):
    env = construct_env('BreakoutNoFrameskip-v4', frame_stack=True,
                        record_video=True, record_video_steps=500, frame_skip=4)

    model = train_model(
        env,
        conv_layers=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        learning_rate=1e-4,
        total_timesteps=int(1e7),
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        batch_size=32,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        checkpoint_path='./checkpoints/',
        double_dqn=use_double_dqn
    )

    model.save('./saved_model/ddqn_breakout/')
    env.close()
