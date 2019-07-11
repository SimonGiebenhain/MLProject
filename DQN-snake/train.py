import pygame
import tensorflow as tf

from actorCritic import ActorCritic
from environment import *
from utils import get_checkpoint_path, get_file_writer



model_name = 'convDQN'
learning_rate = 0.0001
memory_size = 50000
discount_rate = 0.95
eps_min = 0
training_interval = 2
n_steps = 5000000


session = tf.Session()


def train(env, agent):
    file_writer = get_file_writer(model_name=model_name, session=session)
    checkpoint_path = get_checkpoint_path(model_name=model_name)

    running = True
    done = False
    iteration = 0
    n_games = 0
    mean_score = 0

    EMA = 0
    alpha = 0.005

    with session:
        training_start = agent.start(checkpoint_path)

        while running:
            iteration += 1
            if should_display and n_games % 100 == 0:
                env.display()

            if done:  # Game over, start a new game
                n_games += 1
                if n_games == 1:
                    EMA = env.snake.total
                else:
                    EMA = alpha * env.snake.total + (1-alpha) * EMA
                env.reset()
                mean_score = env.total_rewards / n_games

            for event in pygame.event.get():  # Stop the program if we quit the game
                if event.type == pygame.QUIT:
                    running = False

            observation = env.screenshot()
            cur_state = env.get_last_frames(observation)
            step = agent.global_step.eval()

            action, epsilon = agent.act(cur_state, step)
            new_state, reward, done = env.step(action)
            agent.remember(cur_state, action, reward, new_state, done)

            # Only train at regular intervals
            if iteration < training_start or iteration % training_interval != 0:
                continue

            # Train the agent
            agent.train(checkpoint_path, file_writer, mean_score)

            if iteration % 500 == 0:
                print("\rTraining step {}/{} ({:.1f})%\t Record {:.2f} \t Mean score {:.2f} \t EMA {:.2f} \t epsilon {:.2f}".format(
                    step, n_steps, step * 100 / n_steps, env.record, mean_score, EMA, epsilon), end="")

            if step > n_steps:
                break


should_display = True

pygame.init()  # Intializes the game
environment = Environment()
training_agent = ActorCritic(sess=session, training_steps=n_steps, learning_rate=learning_rate,
                             memory_size=memory_size, discount_rate=discount_rate,
                             eps_min=eps_min)
train(environment, training_agent)
