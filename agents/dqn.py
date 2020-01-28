import random
import numpy as np
import gym
import tensorflow as tf
import cv2

from tensorflow.keras import callbacks
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from collections import deque

cv2.ocl.setUseOpenCL(False)
EPISODES = 50000


class ExperienceReplayMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class DQNAgent:
    def __init__(self, action_size, state_size, img_size=84):
        self.render = False
        self.load_model = False

        # Env settings
        self.state_size = state_size
        self.action_size = action_size
        self.img_size = img_size

        # Epsilon parameters
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.0
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps

        # Training parameters
        self.batch_size = 32
        self.initial_replay_size = 15000
        self.train_interval = 4
        self.update_target_rate = 10000
        self.gamma = 0.99
        self.memory = ExperienceReplayMemory(max_size=100000)
        self.no_op_steps = 30

        self.optimizer = Adam(learning_rate=0.00025)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

        self.avg_q_max, self.avg_loss = 0, 0
        self.log_dir = './summary/dqn'
        self.writer = tf.summary.create_file_writer(self.log_dir)

        if self.load_model:
            self.model.load_weights('./saved_model/breakout_dqn.h5')

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (2, 2), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        huber = Huber()
        model.compile(loss=huber,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model

    # After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def add_to_memory(self, history, action, reward, next_history, dead):
        self.memory.add((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = self.memory.sample(batch_size=self.batch_size)

        mb_history = np.zeros((self.batch_size, self.state_size[0],
                               self.state_size[1], self.state_size[2]))
        mb_next_history = np.zeros((self.batch_size, self.state_size[0],
                                    self.state_size[1], self.state_size[2]))

        mb_action, mb_reward, mb_dead = [], [], []

        for i in range(self.batch_size):
            mb_history[i] = np.float32(mini_batch[i][0] / 255.)
            mb_next_history[i] = np.float32(mini_batch[i][3] / 255.)
            mb_action.append(mini_batch[i][1])
            mb_reward.append(mini_batch[i][2])
            mb_dead.append(mini_batch[i][4])

        # actions_mask = np.ones((self.batch_size, self.action_size))
        # target_value = self.target_model.predict([mb_next_history, actions_mask])
        target_value = self.target_model.predict(mb_next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        target_q_mini_batch = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if mb_dead[i]:
                target_q_mini_batch[i] = mb_reward[i]  # maybe -1?
            else:
                target_q_mini_batch[i] = mb_reward[i] + self.gamma * np.amax(target_value[i])

        # action_one_hot = self.get_one_hot(mb_action, self.action_size)
        # target_one_hot = action_one_hot * target_q_mini_batch[:, None]

        # run_history = self.model.fit([mb_history, action_one_hot], target_one_hot,
        #                              epochs=1, batch_size=self.batch_size, verbose=0)
        run_history = self.model.fit(mb_history, target_q_mini_batch,
                                     epochs=1, batch_size=self.batch_size, verbose=0)
        loss = run_history.history['loss'][0]
        self.avg_loss += loss

    def save_model(self, name):
        self.model.save_weights(name)

    @staticmethod
    def get_one_hot(targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    # Input:
    # 210x160x3(colour image)
    # Output: 84x84(mono image)
    # also convert floats to ints to reduce replay memory size
    @staticmethod
    def pre_process(frame):
        return np.uint8(
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA) * 255
        )


if __name__ == '__main__':
    # Create a breakout environment
    # Frameskip of 4 (repeats chosen action 4 times)
    # No random action stochasticity
    # https://github.com/openai/gym/issues/1280#issuecomment-466820285
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3, state_size=(84, 84, 4))

    # tf.summary.trace_on(graph=True, profiler=True)

    tensorboard_callback = callbacks.TensorBoard(log_dir=agent.log_dir)
    # agent.model.predict(np.zeros((1, 84, 84, 4)), callbacks=[tensorboard_callback])

    # # Trace in the graph for showing the model in TensorBoard
    # with agent.writer.as_default():
    #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=agent.log_dir)

    scores, episodes, global_step = [], [], 0

    for i_episode in range(EPISODES):
        is_done = False
        is_dead = False
        # We have 5 lives every episode
        step, score, start_lives = 0, 0, 5

        # Reset the atari environment
        next_frame = env.reset()

        # At the start of the episode we don't have any state information
        # we do nothing to prevent sub-optimal moves, from the deepmind paper
        for _ in range(random.randint(1, agent.no_op_steps)):
            next_frame, _, _, _ = env.step(1)

        # At the start of the episode, there are no preceding frames
        # copy the previous ones to make a history
        processed_frame = agent.pre_process(next_frame)
        frame_history = np.stack((processed_frame, processed_frame, processed_frame, processed_frame), axis=2)
        frame_history = np.reshape([frame_history], (1, 84, 84, 4))

        while not is_done:
            if agent.render:
                env.render()

            global_step += 1
            step += 1

            # get action for the current history and iterate one step
            next_action = agent.get_action(frame_history)

            # convert the action into the format required by OpenAI gym
            action_to_take = min(3, next_action + 1)

            next_frame, next_reward, is_done, info = env.step(action_to_take)

            # Pre-process the frame and add to history
            next_state = agent.pre_process(next_frame)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_iter_history = np.append(next_state, frame_history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict_on_batch(np.float32(frame_history / 255.0))[0]
            )

            # if the agent missed the ball, agent is dead but episode not over
            if start_lives > info['ale.lives']:
                is_dead = True
                start_lives = info['ale.lives']

            clipped_reward = np.clip(next_reward, -1.0, 1.0)

            # Save the <s, a, r, s'> to replay memory
            agent.add_to_memory(frame_history, action_to_take, clipped_reward, next_iter_history, is_dead)

            # Every step, train the model
            if global_step > agent.initial_replay_size:
                # Train network
                if global_step % agent.train_interval == 0:
                    agent.train_replay()

                # Update the target model
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

            score += clipped_reward

            if is_dead:
                is_dead = False
            else:
                frame_history = next_iter_history

            # If the episode is over, plot the score over episodes
            if is_done:
                # with agent.writer.as_default():
                #     with tf.summary.record_if(global_step > agent.train_start):
                #         tf.summary.scalar('Total Reward/Episode', score, step=step)
                #         tf.summary.scalar('Average Max Q/Episode', agent.avg_q_max / float(step), step=step)
                #         tf.summary.scalar('Duration/Episode', i_episode, step=step)
                #         tf.summary.scalar('Average Loss/Episode', agent.avg_loss / float(step), step=step)

                template = 'Episode: {}, Score: {}, Epsilon: {}, Global Steps: {}, Episode Steps: {}, Average Loss: {' \
                           '}, Memory length: {} '
                memory_size = len(agent.memory)
                print(template.format(i_episode,
                                      score,
                                      agent.epsilon,
                                      global_step,
                                      step,
                                      agent.avg_loss / float(step),
                                      memory_size))

                agent.avg_q_max, agent.avg_loss = 0, 0

        if i_episode % 1000 == 0:
            agent.model.save_weights(f'./saved_model/breakout_dqn_{i_episode}.h5')

    env.close()
