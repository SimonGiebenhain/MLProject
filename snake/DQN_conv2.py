# TODO: manchmal geht schlange durch essen durch


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate

from keras.models import Model

import tensorflow as tf

import random
import numpy as np
import pandas as pd
from operator import add
from math import floor
import os


# To be more robust to outliers, we use a quadratic loss for small errors, and a linear loss for large ones.
def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.95 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.dataframe = pd.DataFrame()
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 11
        self.learning_rate = 0.0001
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0

        self.policy_net = self.network()
        #self.target_net = self.network()
        #self.target_net.set_weights(self.policy_net.get_weights())
        #self.policy_net = self.network("weights.hdf5")
        #self.target_net = self.network("weights.hdf5")
        self.memory_size = 50000

        self.epsilon = 0
        self.actual = []
        self.memory_position = 0
        self.memory_state = []
        self.memory_reward = []
        self.memory_action = []
        self.memory_next_state = []
        self.memory_done = []
        self.memory_board = []
        self.memory_next_board = []

    def get_immediate_danger(self, game):
        player = game.player
        x = player.x
        y = player.y
        body = player.position[1:]  # compute danger for next position of body

        danger_straight = 0
        if player.x_change == 20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
            danger_straight = 1
        elif player.x_change == -20 and ([x - 20, y] in body or x - 20 < 20):
            danger_straight = 1
        elif player.y_change == 20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
            danger_straight = 1
        elif player.y_change == -20 and ([x, y - 20] in body or y - 20 < 20):
            danger_straight = 1

        danger_right = 0
        if player.x_change == 20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
            danger_right = 1
        elif player.x_change == -20 and ([x, y - 20] in body or y - 20 < 20):
            danger_right = 1
        elif player.y_change == 20 and ([x - 20, y] in body or x - 20 < 20):
            danger_right = 1
        elif player.y_change == -20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
            danger_right = 1

        danger_left = 0
        if player.x_change == 20 and ([x, y - 20] in body or y - 20 < 20):
            danger_left = 1
        elif player.x_change == -20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
            danger_left = 1
        elif player.y_change == 20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
            danger_left = 1
        elif player.y_change == -20 and ([x - 20, y] in body or x - 20 < 20):
            danger_left = 1

        immediate_danger = [danger_straight, danger_right, danger_left]
        return immediate_danger

    def get_board(self, game, player):
        board = np.zeros([20, 20])
        if not game.crash:
            x = floor(player.x / 20) - 1
            y = floor(player.y / 20) - 1
            board[x, y] = 2
            for pos in player.position[:-1]:
                body_x = floor(pos[0] / 20) - 1
                bod_y = floor(pos[1] / 20) - 1
                if body_x != x or bod_y != y:
                    board[body_x, bod_y] = 3
            x = floor(game.food.x_food / 20) - 1
            y = floor(game.food.y_food / 20) - 1
            board[x,y] = 1
        return board

        # TODO: work with body position of next state instead
    def get_state(self, game):
            player = game.player
            food = game.food

            state = self.get_immediate_danger(game)

            state.append(player.x_change == -20)  # move left
            state.append(player.x_change == 20)  # move right
            state.append(player.y_change == -20)  # move up
            state.append(player.y_change == 20)  # move down
            state.append(food.x_food < player.x)  # food left
            state.append(food.x_food > player.x)  # food right
            state.append(food.y_food < player.y)  # food up
            state.append(food.y_food > player.y)  # food down

            for i in range(len(state)):
                if state[i]:
                    state[i] = 1
                else:
                    state[i] = 0

            # state.append(player.consecutive_right_turns)
            # state.append(player.consecutive_left_turns)
            # state.append(player.consecutive_straight_before_turn)
            # state.append(game.player.food/game.game_width)

            # state.append(player.x/game.game_width)
            # state.append(player.y/game.game_height)
            # state.append((food.x_food - player.x) / game.game_width),  # food x difference
            # state.append((food.y_food - player.y) / game.game_height)  # food y difference

            # state.append(self.last_move[1]*self.move_count/10)
            # state.append(self.last_move[2]*self.move_count/10)

            # TODO:
            # add length of snake to state
            # state.append(player.food/game.game_width)

            # calculate distances to next wall in each direction as additional information
            #if player.x_change == -20:
            #    d_wall_straight = player.position[-1][0] / game.game_width
            #    d_wall_backwards = (game.game_width - player.position[-1][0]) / game.game_width
            #    d_wall_right = player.position[-1][1] / game.game_height
            #    d_wall_left = (game.game_height - player.position[-1][1]) / game.game_height
            ##
            #elif player.x_change == 20:
            #    d_wall_straight = (game.game_width - player.position[-1][0]) / game.game_width
            #    d_wall_backwards = player.position[-1][0] / game.game_width
            #    d_wall_right = (game.game_height - player.position[-1][1]) / game.game_height
            #    d_wall_left = player.position[-1][1] / game.game_height
            ##
            #elif player.y_change == -20:
            #    d_wall_straight = player.position[-1][1] / game.game_height
            #    d_wall_backwards = (game.game_height - player.position[-1][1]) / game.game_height
            #    d_wall_right = (game.game_width - player.position[-1][0]) / game.game_width
            #    d_wall_left = player.position[-1][0] / game.game_width
            ##
            #else:
            #    d_wall_straight = (game.game_height - player.position[-1][1]) / game.game_height
            #    d_wall_backwards = player.position[-1][1] / game.game_height
            #    d_wall_right = player.position[-1][0] / game.game_width
            #    d_wall_left = (game.game_width - player.position[-1][0]) / game.game_width
            ##
            ##
            ## calculate distances to own body, if none than use distance to next wall
            #if player.x_change == -20:
            #    x = player.position[-1][0]
            #    y = player.position[-1][1]
            #    #
            #    # straight
            #    candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
            #    if candidates:
            #        closest = max(candidates)
            #    else:
            #        closest = 0
            #    d_body_straight = (x - closest) / game.game_width
            #    #
            #    # right
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
            #    if candidates:
            #        closest = max(candidates)
            #    else:
            #        closest = 0
            #    d_body_right = (y - closest) / game.game_height
            #    #
            #    # left
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            #    if candidates:
            #        closest = min(candidates)
            #    else:
            #        closest = game.game_height - 20
            #    d_body_left = (closest - y) / game.game_height
            ##
            ##
            #elif player.x_change == 20:
            #    x = player.position[-1][0]
            #    y = player.position[-1][1]
            #    #
            #    # straight
            #    candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
            #    if candidates:
            #        closest = min(candidates)
            #    else:
            #        closest = game.game_width - 20
            #    d_body_straight = (closest - x) / game.game_width
            #    #
            #    # right
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            #    if candidates:
            #        closest = min(candidates)
            #    else:
            #        closest = game.game_height - 20
            #    d_body_right = (closest - y) / game.game_height
            #    #
            #    # left
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
            #    if candidates:
            #        closest = max(candidates)
            #    else:
            #        closest = 0
            #    d_body_left = (y - closest) / game.game_height
            ##
            ##
            #elif player.y_change == -20:
            #    x = player.position[-1][0]
            #    y = player.position[-1][1]
            #    #
            #    # straight
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
            #    if candidates:
            #        closest = max(candidates)
            #    else:
            #        closest = 0
            #    d_body_straight = (y - closest) / game.game_height
            #    #
            #    # right
            #    candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
            #    if candidates:
            #        closest = min(candidates)
            #    else:
            #        closest = game.game_width - 20
            #    d_body_right = (closest - x) / game.game_width
            #    #
            #    # left
            #    candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
            #    if candidates:
            #        closest = max(candidates)
            #    else:
            #        closest = 0
            #    d_body_left = (x - closest) / game.game_width
            ##
            ##
            ## player.y_change == 20:
            #else:
            #    x = player.position[-1][0]
            #    y = player.position[-1][1]
            #    #
            #    # straight
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            #    if candidates:
            #        closest = min(candidates)
            #    else:
            #        closest = game.game_height - 20
            #    d_body_straight = (closest - y) / game.game_height
            #    #
            #    # right
            #    candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
            #    if candidates:
            #        closest = max(candidates)
            #    else:
            #        closest = 0
            #    d_body_right = (x - closest) / game.game_width
            #    #
            #    # left
            #    candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            #    if candidates:
            #        closest = min(candidates)
            #    else:
            #        closest = game.game_width - 20
            #    d_body_left = (closest - x) / game.game_width
            #
            #state.append(d_body_straight)
            #state.append(d_body_left)
            #state.append(d_body_right)
            # state.append(min([d_wall_right, d_body_right]))
            # state.append(min([d_wall_straight, d_body_straight]))
            #state.append(d_wall_backwards)
            # state.append(min([d_wall_left, d_body_left]))
            #state.append(d_wall_straight)
            #state.append(d_wall_right)
            #state.append(d_wall_left)

            # TODO: use more rays
            # TODO: Try out distance to free space
            # TODO: lönge ja oder nein oder als kurz mittel und lange oder sowas?

            board = self.get_board(game, player)
            # code = self.encoder.predict((np.expand_dims(board,0)))
            # board_lin = np.reshape(board, -1)

            # return np.expand_dims(np.asarray(state), 0)
            # return np.concatenate([ np.asarray(state), board_lin])
            return np.asarray(state), board


    def set_reward(self, game, player, crash, crash_reason, curr_move, state_old, steps):

        self.reward = 0
        if crash:
            self.reward = -10
        elif player.eaten:
            self.reward = 10 #5 + player.food/10
        else:
            # self.reward = -0.01
            if steps > player.food * 1.2 + 15:
                self.reward = - 0.5 / player.food

        return self.reward


    def remember(self, state, board, action, reward, next_state, next_board, done):
        if len(self.memory_done) < self.memory_size:
            self.memory_state.append(state)
            self.memory_action.append(action)
            self.memory_reward.append(reward)
            self.memory_next_state.append(next_state)
            self.memory_done.append(done)
            self.memory_board.append(board)
            self.memory_next_board.append(next_board)
        else:
            self.memory_state[self.memory_position] = state
            self.memory_action[self.memory_position] = action
            self.memory_reward[self.memory_position] = reward
            self.memory_next_state[self.memory_position] = next_state
            self.memory_done[self.memory_position] = done
            self.memory_board[self.memory_position] = board
            self.memory_next_board[self.memory_position] = next_board

            self.memory_position = (self.memory_position + 1) % self.memory_size



    def network(self, weights=None):

        num_inp = Input(shape=[self.state_length])
        num_feats = Dense(120, activation='relu')(num_inp)

        board_inp = Input(shape=[20,20,2])

        board_feats = BatchNormalization()(Conv2D(8, kernel_size=(5,5), strides=(2,2), activation='relu')(board_inp))

        board_feats = BatchNormalization()(Conv2D(16, kernel_size=(3,3), strides=(2,2), activation='relu')(board_feats))

        board_feats = Flatten()(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu')(board_feats))

        board_feats = Dense(350, activation='relu')(board_feats)
        board_feats = Dense(100, activation='relu')(board_feats)
        feats = Concatenate()([num_feats, board_feats])
        feats = Dense(150, activation='relu')(feats)
        feats = Dense(50, activation='relu')(feats)
        output = Dense(3)(feats)

        model = Model([num_inp, board_inp], output)
        model.summary()
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def replay_new_vectorized(self):
       batch_size = 128
       if len(self.memory_done) > batch_size:
           rng_state = random.getstate()
           state_minibatch = np.squeeze(np.asarray(random.sample(self.memory_state, batch_size)))
           random.setstate(rng_state)
           board_minibatch = np.squeeze(np.asarray(random.sample(self.memory_board, batch_size)))
           random.setstate(rng_state)
           action_minibatch = np.asarray(random.sample(self.memory_action, batch_size))
           random.setstate(rng_state)
           reward_minibatch = np.asarray(random.sample(self.memory_reward, batch_size))
           random.setstate(rng_state)
           next_state_minibatch = np.squeeze(np.asarray(random.sample(self.memory_next_state, batch_size)))
           random.setstate(rng_state)
           next_board_minibatch = np.squeeze(np.asarray(random.sample(self.memory_next_board, batch_size)))
           random.setstate(rng_state)
           done_minibatch = np.asarray(random.sample(self.memory_done, batch_size))
       else:
           state_minibatch = np.squeeze(np.asarray(self.memory_state))
           board_minibatch = np.squeeze(np.asarray(self.memory_board))
           action_minibatch = np.asarray(self.memory_action)
           reward_minibatch = np.asarray(self.memory_reward)
           next_state_minibatch = np.squeeze(np.asarray(self.memory_next_state))
           next_board_minibatch = np.squeeze(np.asarray(self.memory_next_board))
           done_minibatch = np.asarray(self.memory_done)

       target = reward_minibatch
       target[np.invert(done_minibatch)] = target[np.invert(done_minibatch)] + self.gamma * np.amax(self.policy_net.predict([next_state_minibatch, next_board_minibatch]), 1)[np.invert(done_minibatch)]
       target_f = self.policy_net.predict([state_minibatch, board_minibatch])
       target_f[:, np.argmax(action_minibatch, 1)] = target
       self.policy_net.fit([state_minibatch, board_minibatch],target_f, epochs=1, verbose=0)


