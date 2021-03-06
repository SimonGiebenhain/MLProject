# TODO: manchmal geht schlange durch essen durch

from random import randint, choice
import copy
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate

from keras.models import Model

from keras.utils import to_categorical


import random
import numpy as np
import pandas as pd
from operator import add
from math import floor


class Game:

    def __init__(self, game_width, game_height, x, y, body, x_change, y_change, food, food_x, food_y):
        self.game_width = game_width
        self.game_height = game_height
        self.crash = False
        self.player = Player(x, y, body, x_change, y_change, food)
        self.food = Food(food_x, food_y)
        self.score = 0

class Player(object):

    def __init__(self, x, y, body, x_change, y_change, food):
        self.x = x
        self.y = y
        self.position = copy.deepcopy(body)
        self.food = food
        self.eaten = False
        self.x_change = x_change
        self.y_change = y_change

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, x, y, game, food):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change
        self.update_position(self.x, self.y)


        if self.x < 20 or self.x > game.game_width - 40 or self.y < 20 or self.y > game.game_height - 40:
            game.crash = True
            game.crash_reason = 0
        elif [self.x, self.y] in self.position[:-2]:
            game.crash = True
            game.crash_reason = 10


        if eat(self, food, game):
            game.crash = True
            game.score = 1000000000000

class Food(object):

    def __init__(self, x, y):
        self.x_food = x
        self.y_food = y

    def food_coord(self, game, player):
        possible_location = []
        for i in range(20, game.game_width - 20, 20):
            for j in range(20, game.game_height - 20, 20):
                possible_location.append([i, j])
        for body_part in game.player.position:
            if body_part in possible_location:
                possible_location.remove(body_part)
        if len(possible_location) == 0:
            return True
        food = choice(possible_location)
        self.x_food = food[0]
        self.y_food = food[1]
        return False


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        if food.food_coord(game, player):
            return True
        player.eaten = True
        game.score += 1
        return False



class DQNAgent(object):

    def __init__(self):
        self.trainable = False
        self.type = 'DQNSimpleMCAgent'
        self.reward = 0
        self.gamma = 0.90 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 25
        #self.code_length = 90
        self.learning_rate = 0.0005
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        self.policy_net = self.network()
        self.target_net = self.network()
        #self.target_net.set_weights(self.policy_net.get_weights())
        self.policy_net = self.network("weights/vicinity10_weights.hdf5")
        self.target_net = self.network("weights/vicinity10_weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []
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
        body = player.position[1:] # compute danger for next position of body

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

        body = body[1:] # all dangers to come lie 2 steps into the future

        danger_right_straight = 0
        if player.x_change == 20 and ([x + 20, y + 20] in body or y + 20 >= game.game_height - 20  or x + 20 >= game.game_height - 20):
            danger_right_straight = 1
        elif player.x_change == -20 and ([x - 20, y - 20] in body or y - 20 < 20 or x - 20 < 20):
            danger_right_straight = 1
        elif player.y_change == 20 and ([x - 20, y + 20] in body or x - 20 < 20 or y + 20 >= game.game_width - 20):
            danger_right_straight = 1
        elif player.y_change == -20 and ([x + 20, y - 20] in body or x + 20 >= game.game_height - 20 or y - 20 < 20):
            danger_right_straight = 1

        danger_left_straight = 0
        if player.x_change == 20 and (
                [x + 20, y - 20] in body or y - 20 < 20 or x + 20 >= game.game_height - 20):
            danger_left_straight = 1
        elif player.x_change == -20 and ([x - 20, y + 20] in body or y + 20 >= game.game_width - 20 or x - 20 < 20):
            danger_left_straight = 1
        elif player.y_change == 20 and ([x + 20, y + 20] in body or x + 20 >= game.game_width - 20 or y + 20 >= game.game_width - 20):
            danger_left_straight = 1
        elif player.y_change == -20 and ([x - 20, y - 20] in body or x - 20 < 20 or y - 20 < 20):
            danger_left_straight = 1

        danger_straight_straight = 0
        if player.x_change == 20 and ([x + 40, y] in body or x + 40 >= game.game_width - 20):
            danger_straight_straight = 1
        elif player.x_change == -20 and ([x - 40, y] in body or x - 40 < 20):
            danger_straight_straight = 1
        elif player.y_change == 20 and ([x, y + 40] in body or y + 40 >= game.game_height - 20):
            danger_straight_straight = 1
        elif player.y_change == -20 and ([x, y - 40] in body or y - 40 < 20):
            danger_straight_straight = 1

        danger_right_right = 0
        if player.x_change == 20 and ([x, y + 40] in body or y + 40 >= game.game_height - 20):
            danger_right_right = 1
        elif player.x_change == -20 and ([x, y - 40] in body or y - 40 < 20):
            danger_right_right = 1
        elif player.y_change == 20 and ([x - 40, y] in body or x - 40 < 20):
            danger_right_right = 1
        elif player.y_change == -20 and ([x + 40, y] in body or x + 40 >= game.game_width - 20):
            danger_right_right = 1

        danger_left_left = 0
        if player.x_change == 20 and ([x, y - 40] in body or y - 40 < 20):
            danger_left_left = 1
        elif player.x_change == -20 and ([x, y + 40] in body or y + 40 >= game.game_height - 20):
            danger_left_left = 1
        elif player.y_change == 20 and ([x + 40, y] in body or x + 40 >= game.game_width - 20):
            danger_left_left = 1
        elif player.y_change == -20 and ([x - 40, y] in body or x - 40 < 20):
            danger_left_left = 1

        danger_right_back = 0
        if player.x_change == 20 and ([x - 20, y + 20] in body or y + 20 >= game.game_height - 20 or x - 20 < 20):
            danger_right_back = 1
        elif player.x_change == -20 and ([x + 20, y - 20] in body or y - 20 < 20 or x + 20 >= game.game_width - 20):
            danger_right_back = 1
        elif player.y_change == 20 and ([x - 20, y - 20] in body or x - 20 < 20 or y - 20 < 20):
            danger_right_back = 1
        elif player.y_change == -20 and ([x + 20, y + 20] in body or x + 20 >= game.game_width - 20 or y + 20 >= game.game_height - 20):
            danger_right_back = 1

        danger_left_back = 0
        if player.x_change == 20 and ([x - 20, y - 20] in body or y - 20 < 20 or x - 20 < 20):
            danger_left_back = 1
        elif player.x_change == -20 and ([x + 20, y + 20] in body or y + 20 >= game.game_height - 20  or x + 20 >= game.game_width - 20):
            danger_left_back = 1
        elif player.y_change == 20 and ([x + 20, y - 20] in body or x + 20 >= game.game_width - 20 or y - 20 < 20):
            danger_left_back = 1
        elif player.y_change == -20 and ([x - 20, y + 20] in body or x - 20 < 20 or y + 20 >= game.game_height - 20):
            danger_left_back = 1

        immediate_danger = [danger_straight, danger_right, danger_left, danger_right_straight, danger_left_straight, danger_straight_straight, danger_right_right, danger_left_left, danger_right_back, danger_left_back]
        return immediate_danger

    def get_board(self, game, player):
        board = np.zeros([20, 20, 4])
        board[:,:,0] = 1
        if not game.crash:
            x = floor(player.x / 20) - 1
            y = floor(player.y / 20) - 1
            board[x,y,3] = 1
            board[x,y,0] = 0
            for pos in player.position[:-2]:
                x = floor(pos[0] / 20) - 1
                y = floor(pos[1] / 20) - 1
                board[x, y, 0] = 0
                board[x,y, 2] = 1
            x = floor(game.food.x_food / 20) - 1
            y = floor(game.food.y_food / 20) - 1
            if board[x,y,3] == 0:
                board[x,y,1] = 1
                board[x,y,0] = 0
        return board


    # TODO: work with body position of next state instead
    def get_state(self, game):
       player = game.player
       food = game.food

       state = self.get_immediate_danger(game)
       danger = state[:3]

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
       if player.x_change == -20:
           d_wall_straight = player.position[-1][0] / game.game_width
           d_wall_backwards = (game.game_width - player.position[-1][0]) / game.game_width
           d_wall_right = player.position[-1][1] / game.game_height
           d_wall_left = (game.game_height - player.position[-1][1]) / game.game_height
       #
       elif player.x_change == 20:
           d_wall_straight = (game.game_width - player.position[-1][0]) / game.game_width
           d_wall_backwards = player.position[-1][0] / game.game_width
           d_wall_right = (game.game_height - player.position[-1][1]) / game.game_height
           d_wall_left = player.position[-1][1] / game.game_height
       #
       elif player.y_change == -20:
           d_wall_straight = player.position[-1][1] / game.game_height
           d_wall_backwards = (game.game_height - player.position[-1][1]) / game.game_height
           d_wall_right = (game.game_width - player.position[-1][0]) / game.game_width
           d_wall_left = player.position[-1][0] / game.game_width
       #
       else:
           d_wall_straight = (game.game_height - player.position[-1][1]) / game.game_height
           d_wall_backwards = player.position[-1][1] / game.game_height
           d_wall_right = player.position[-1][0] / game.game_width
           d_wall_left = (game.game_width - player.position[-1][0]) / game.game_width
       #
       #
       # calculate distances to own body, if none than use distance to next wall
       if player.x_change == -20:
           x = player.position[-1][0]
           y = player.position[-1][1]
           #
           # straight
           candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
           if candidates:
               closest = max(candidates)
           else:
               closest = 0
           d_body_straight = (x - closest) / game.game_width
           #
           # right
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
           if candidates:
               closest = max(candidates)
           else:
               closest = 0
           d_body_right = (y - closest) / game.game_height
           #
           # left
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
           if candidates:
               closest = min(candidates)
           else:
               closest = game.game_height - 20
           d_body_left = (closest - y) / game.game_height
       #
       #
       elif player.x_change == 20:
           x = player.position[-1][0]
           y = player.position[-1][1]
           #
           # straight
           candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
           if candidates:
               closest = min(candidates)
           else:
               closest = game.game_width - 20
           d_body_straight = (closest - x) / game.game_width
           #
           # right
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
           if candidates:
               closest = min(candidates)
           else:
               closest = game.game_height - 20
           d_body_right = (closest - y) / game.game_height
           #
           # left
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
           if candidates:
               closest = max(candidates)
           else:
               closest = 0
           d_body_left = (y - closest) / game.game_height
       #
       #
       elif player.y_change == -20:
           x = player.position[-1][0]
           y = player.position[-1][1]
           #
           # straight
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
           if candidates:
               closest = max(candidates)
           else:
               closest = 0
           d_body_straight = (y - closest) / game.game_height
           #
           # right
           candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
           if candidates:
               closest = min(candidates)
           else:
               closest = game.game_width - 20
           d_body_right = (closest - x) / game.game_width
           #
           # left
           candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
           if candidates:
               closest = max(candidates)
           else:
               closest = 0
           d_body_left = (x - closest) / game.game_width
       #
       #
       # player.y_change == 20:
       else:
           x = player.position[-1][0]
           y = player.position[-1][1]
           #
           # straight
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
           if candidates:
               closest = min(candidates)
           else:
               closest = game.game_height - 20
           d_body_straight = (closest - y) / game.game_height
           #
           # right
           candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
           if candidates:
               closest = max(candidates)
           else:
               closest = 0
           d_body_right = (x - closest) / game.game_width
           #
           # left
           candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
           if candidates:
               closest = min(candidates)
           else:
               closest = game.game_width - 20
           d_body_left = (closest - x) / game.game_width
       #
       state.append(d_body_straight)
       state.append(d_body_left)
       state.append(d_body_right)
       # state.append(min([d_wall_right, d_body_right]))
       # state.append(min([d_wall_straight, d_body_straight]))
       state.append(d_wall_backwards)
       # state.append(min([d_wall_left, d_body_left]))
       state.append(d_wall_straight)
       state.append(d_wall_right)
       state.append(d_wall_left)

       # TODO: use more rays
       # TODO: Try out distance to free space
       # TODO: lönge ja oder nein oder als kurz mittel und lange oder sowas?

       # board = self.get_board(game, player)
       # code = self.encoder.predict((np.expand_dims(board,0)))
       # board_lin = np.reshape(board, -1)

       # return np.expand_dims(np.asarray(state), 0)
       # return np.concatenate([ np.asarray(state), board_lin])
       return {'state': np.asarray(state), 'danger': danger}  # , code

    # check whether snake is going straight into slot of width 1


    def set_reward(self, game, player, crash, crash_reason, curr_move, state_old, steps):

        #TODO:
        #       - sind viele kurven wirklich schlecht? immerhin kann es eine kompakte schlange geben
        #       - check for closed loops in reward and give -100 or smth.
        #       - viel platz verbrauchen bestrafen, zb. einzelne spalte oder zeile frei lassen ist nicht gut (oder ungerade anzahl)
        #           - oder halt belohnen wenn sich die schlange berührt
        #       - was noch?
        self.reward = 0
        if crash:
            self.reward = -10
            #if not (state_old[0] == 1 and state_old[1] == 1 and state_old[2] == 1):
            #    self.reward = -10 #- crash_reason
            #else:
            #    self.reward = -5
            #return self.reward
        elif player.eaten:
            self.reward = 10 #5 + player.food/10
        else:
            #self.reward = -0.01
            if steps > player.food * 1.2 + 15:
                self.reward = - 0.5 / player.food

        #elif self.last_move == 1:
        #    self.reward += -0.03
        #TODO: wenn keine andere wahl don't punish!!
        # going in circles
        #if player.food > 10:
        #    if self.move_count >= 3 and self.did_turn:
        #        self.reward -= self.move_count/5
                #print('move count: ', self.move_count)
        # go towards food, else get reckt
        #if player.food_distance < player.food_distance_old:
        #    self.reward += 0.1
        #else:
        #    self.reward -= 0.002

        # punish going into slot of width 1
        #if player.food > 10 and not self.did_turn:
        #    self.straight_sackgasse(game, player)

        #if player.food > 10 and self.did_turn:
        #    self.punish_loop(game, player, curr_move)

        return self.reward


    # TODO try conv net
    def network(self, weights=None):

        num_inp = Input(shape=[self.state_length])
        x = Dense(100, activation='relu')(num_inp)

        #model.add(Dropout(0.15))
        x = Dense(120, activation='relu')(x)
        #model.add(Dropout(0.1))
        x = Dense(50, activation='relu')(x)
        #model.add(Dropout(0.05))
        output = Dense(3)(x)

        model = Model(num_inp, output)
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def rollout(self, game, action, eps, max_steps):

        body = game.player.position
        # max sim steps
        simulation = Game(game.game_width, game.game_height, game.player.x, game.player.y, body,
                                     game.player.x_change, game.player.y_change, game.player.food, game.food.x_food, game.food.y_food)

        simulation.player.do_move(action, game.player.x, game.player.y, simulation, simulation.food)

        simulation_steps = 0
        while not simulation.crash and simulation_steps < max_steps:
            simulation_steps += 1
            state = self.get_state(simulation)['state']

            if np.random.rand() < eps:
                possible_action = []
                if state[0] == 0:
                    possible_action.append([1, 0, 0])
                if state[1] == 0:
                    possible_action.append([0, 1, 0])
                if state[2] == 0:
                    possible_action.append([0, 0, 1])
                if len(possible_action) > 0:
                    action = choice(possible_action)
                else:
                    action = choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            else:
                prediction = self.policy_net.predict(np.reshape(state, [1, self.state_length]))
                action = to_categorical(np.argmax(prediction[0]), num_classes=3)

            simulation.player.do_move(action, simulation.player.x, simulation.player.y, simulation, simulation.food)

        if not simulation.crash:
            # TODO better heursitic or ask policynet or train value network
            return 1*min(1,simulation.score) + 50 + 0.2*simulation.score
            #return 0.1*simulation.score + 20
        else:
            return  1* min(1,simulation.score) + 1.8*simulation_steps/max_steps + 0.2*simulation.score
            #return 0.1*simulation.score + 2*simulation_steps/max_steps


    def act(self, game, state, state_old, action_old):

        # TODO never go in sackgassse
        num_simulations = 1
        eps_simulation = 0
        max_steps = 50

        danger_straight = state[0]
        danger_right = state[1]
        danger_left = state[2]
        danger_straight_right = state[3]
        danger_straight_left = state[4]
        if np.array_equal(action_old, [1, 0, 0]):
            danger_straight_right_old = state_old[3]
            danger_straight_left_old = state_old[4]
        else:
            danger_straight_right_old = 0
            danger_straight_left_old = 0
        danger_straight_old = state_old[0]
        danger_right_old = state_old[1]
        danger_left_old = state_old[2]


        if (game.score > 75 or ((danger_straight == 1  or (danger_straight_right == 1 and danger_straight_right_old == 0) or (danger_straight_left == 1 and danger_straight_left_old == 0)))) and danger_straight + danger_right + danger_left < 2:
            #print('Running simulations')

            avg_reward_straight = -1000
            if danger_straight == 0:
                action = [1, 0, 0]
                avg_reward_straight = 0
                for i in range(num_simulations):
                    avg_reward_straight += self.rollout(game, action, eps_simulation, max_steps)
                avg_reward_straight = avg_reward_straight / num_simulations

            avg_reward_right = -1000
            if danger_right == 0:
                action = [0, 1, 0]
                avg_reward_right = 0
                for i in range(num_simulations):
                    avg_reward_right += self.rollout(game, action, eps_simulation, max_steps)
                avg_reward_right = avg_reward_right / num_simulations

            avg_reward_left = -1000
            if danger_left == 0:
                action = [0, 0, 1]
                avg_reward_left = 0
                for i in range(num_simulations):
                    avg_reward_left += self.rollout(game, action, eps_simulation, max_steps)
                avg_reward_left = avg_reward_left / num_simulations

            avg_rewards = [avg_reward_straight, avg_reward_right, avg_reward_left]
            arr = np.array(avg_rewards)
            #print(avg_rewards)
            maxi = max(avg_rewards)
            possible_actions = [ i for i,rew in enumerate(avg_rewards) if rew == maxi]
            if len(possible_actions) > 2:
                argmax = choice(possible_actions)
            else:
                argmax = avg_rewards.index(maxi)
            action = np.array([0, 0, 0])
            action[argmax] = 1
            return action
        #elif (danger_straight_old + danger_left_old == 2 or danger_straight_old + danger_right_old == 2) and (danger_straight + danger_right + danger_left < 2):
        #    if danger_right == 0:
        #        action = [0, 1, 0]
        #    else: # means danger_left == 0
        #        action = [0, 0, 1]
        #    return action
        else:
            #if np.random.rand() < 0.1:
            #    possible_action = []
            #    if state[0] == 0:
            #        possible_action.append([1, 0, 0])
            #    if state[1] == 0:
            #        possible_action.append([0, 1, 0])
            #    if state[2] == 0:
            #        possible_action.append([0, 0, 1])
            #    if len(possible_action) > 0:
            #        action = choice(possible_action)
            #    else:
            #        action = choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            #    return(action)
            #else:
            prediction = self.policy_net.predict(np.expand_dims(state, 0))
            return to_categorical(np.argmax(prediction[0]), num_classes=3)


