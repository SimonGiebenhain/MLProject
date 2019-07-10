# TODO: manchmal geht schlange durch essen durch


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate

from keras.models import Model

import random
import numpy as np
import pandas as pd
from operator import add
from math import floor


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.95 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 13
        self.code_length = 36
        self.learning_rate = 0.0005
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        self.policy_net = self.network()
        self.target_net = self.network()
        #self.target_net.set_weights(self.policy_net.get_weights())
        self.policy_net = self.network("weights.hdf5")
        self.target_net = self.network("weights.hdf5")
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
        input_img = Input(shape=(20, 20, 2))
        code = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        code = MaxPooling2D((2, 2), padding='same')(code)
        code = Conv2D(8, (3, 3), activation='relu', padding='same')(code)
        code = MaxPooling2D((2, 2), padding='same')(code)
        code = Conv2D(4, (3, 3), activation='relu', padding='same')(code)
        code = MaxPooling2D((2, 2), padding='same')(code)
        code = Flatten()(code)
        encoder = Model(input_img, code)
        encoder.load_weights('encoder_weights.hdf5')
        self.encoder = encoder


    def get_immediate_danger(self, game, player):
        x = player.x
        y = player.y
        body = player.position

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

        return danger_straight, danger_right, danger_left

    def get_board(self, game, player):
        board = np.zeros([20, 20, 2])
        if not game.crash:
            x = floor(player.x / 20) - 1
            y = floor(player.y / 20) - 1
            board[x,y,1] = 1
            for pos in player.position:
                x = floor(pos[0] / 20) - 1
                y = floor(pos[1] / 20) - 1
                board[x, y, 0] = 1
            x = floor(game.food.x_food / 20) - 1
            y = floor(game.food.y_food / 20) - 1
            if board[x,y,0] == 0:
                board[x,y,1] = 1
        return board


    # TODO: work with body position of next state instead
    def get_state(self, game, player, food):
        state = [

            (player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [20, 0])) in player.position) or
                    player.position[-1][0] + 20 >= (game.game_width - 20))) or (
                    player.x_change == -20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [-20, 0])) in player.position) or
                    player.position[-1][0] - 20 < 20)) or (
                    player.x_change == 0 and player.y_change == -20 and (
                    (list(map(add, player.position[-1], [0, -20])) in player.position) or
                    player.position[-1][-1] - 20 < 20)) or (
                    player.x_change == 0 and player.y_change == 20 and (
                    (list(map(add, player.position[-1], [0, 20])) in player.position) or
                    player.position[-1][-1] + 20 >= (game.game_height - 20))),


            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
                                                                  player.position[ -1][0] + 20 >= (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
                                                                                                                                                                                           [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
                add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
                -1] + 20 >= (game.game_height-20))),  # danger right

            (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
                                                                 player.position[-1][0] + 20 >= (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
                add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
                    player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
                                                                         player.position[-1][-1] + 20 >= (game.game_height-20))), #danger left


            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        state.append(player.x/game.game_width)
        state.append(player.y/game.game_height)
        #state.append((food.x_food - player.x) / game.game_width),  # food x difference
        #state.append((food.y_food - player.y) / game.game_height)  # food y difference

        #state.append(self.last_move[1]*self.move_count/10)
        #state.append(self.last_move[2]*self.move_count/10)

        # TODO:
        # add length of snake to state
        #state.append(player.food/game.game_width)

        # calculate distances to next wall in each direction as additional information
#        if player.x_change == -20:
#            d_wall_straight = player.position[-1][0] / game.game_width
#            d_wall_backwards = (game.game_width - player.position[-1][0]) / game.game_width
#            d_wall_right = player.position[-1][1] / game.game_height
#            d_wall_left = (game.game_height - player.position[-1][1]) / game.game_height
#
#        elif player.x_change == 20:
#            d_wall_straight = (game.game_width - player.position[-1][0]) / game.game_width
#            d_wall_backwards = player.position[-1][0] / game.game_width
#            d_wall_right = (game.game_height - player.position[-1][1]) / game.game_height
#            d_wall_left = player.position[-1][1] / game.game_height
#
#        elif player.y_change == -20:
#            d_wall_straight = player.position[-1][1] / game.game_height
#            d_wall_backwards = (game.game_height - player.position[-1][1]) / game.game_height
#            d_wall_right = (game.game_width - player.position[-1][0]) / game.game_width
#            d_wall_left = player.position[-1][0] / game.game_width
#
#        else:
#            d_wall_straight = (game.game_height - player.position[-1][1]) / game.game_height
#            d_wall_backwards = player.position[-1][1] / game.game_height
#            d_wall_right = player.position[-1][0] / game.game_width
#            d_wall_left = (game.game_width - player.position[-1][0]) / game.game_width
#
#
#        # calculate distances to own body, if none than use distance to next wall
#        if player.x_change == -20:
#            x = player.position[-1][0]
#            y = player.position[-1][1]
#
#            # straight
#            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
#            if candidates:
#                closest = max( candidates )
#            else:
#                closest = 0
#            d_body_straight = (x - closest) / game.game_width
#
#            # right
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
#            if candidates:
#                closest = max(candidates)
#            else:
#                closest = 0
#            d_body_right = (y - closest) / game.game_height
#
#            # left
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
#            if candidates:
#                closest = min(candidates)
#            else:
#                closest = game.game_height - 20
#            d_body_left = (closest - y) / game.game_height
#
#
#        elif player.x_change == 20:
#            x = player.position[-1][0]
#            y = player.position[-1][1]
#
#            # straight
#            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
#            if candidates:
#                closest = min(candidates)
#            else:
#                closest = game.game_width - 20
#            d_body_straight = (closest - x) / game.game_width
#
#            # right
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
#            if candidates:
#                closest = min(candidates)
#            else:
#                closest = game.game_height - 20
#            d_body_right = (closest - y) / game.game_height
#
#            # left
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
#            if candidates:
#                closest = max(candidates)
#            else:
#                closest = 0
#            d_body_left = (y - closest) / game.game_height
#
#
#        elif player.y_change == -20:
#            x = player.position[-1][0]
#            y = player.position[-1][1]
#
#            # straight
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
#            if candidates:
#                closest = max(candidates)
#            else:
#                closest = 0
#            d_body_straight = (y - closest) / game.game_height
#
#            # right
#            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
#            if candidates:
#                closest = min(candidates)
#            else:
#                closest = game.game_width - 20
#            d_body_right = (closest - x) / game.game_width
#
#            # left
#            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
#            if candidates:
#                closest = max(candidates)
#            else:
#                closest = 0
#            d_body_left = (x - closest) / game.game_width
#
#
#            #player.y_change == 20:
#        else:
#            x = player.position[-1][0]
#            y = player.position[-1][1]
#
#            # straight
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
#            if candidates:
#                closest = min(candidates)
#            else:
#                closest = game.game_height - 20
#            d_body_straight = (closest - y) / game.game_height
#
#            # right
#            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
#            if candidates:
#                closest = max(candidates)
#            else:
#                closest = 0
#            d_body_right = (x - closest) / game.game_width
#
#            # left
#            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
#            if candidates:
#                closest = min(candidates)
#            else:
#                closest = game.game_width - 20
#            d_body_left = (closest - x) / game.game_width
#
#        state.append(d_body_straight)
#        state.append(d_body_left)
#        state.append(d_body_right)
#        state.append(d_wall_right)
#        state.append(d_wall_straight)
#        state.append(d_wall_backwards)
#        state.append(d_wall_left)


        # TODO: use more rays
        # TODO: Try out distance to free space
        # TODO: lönge ja oder nein oder als kurz mittel und lange oder sowas?

        board = self.get_board(game, player)
        code = self.encoder.predict((np.expand_dims(board,0)))
        #board_lin = np.reshape(board, -1)

        #return np.expand_dims(np.asarray(state), 0)
        #return np.concatenate([ np.asarray(state), board_lin])
        return np.asarray(state), code


    # check whether snake is going straight into slot of width 1
    def straight_sackgasse(self, game, player):
        x = player.x
        y = player.y
        body = player.position
        if player.x_change == 20:
            if (([x, y + 20] in body) or (y + 20 >= game.game_height)) and \
                    (([x, y - 20] in body) or (y - 20 <= 0)) and \
                    (([pos[0] for pos in body[:-2] if pos[1] == y and pos[0] > x]) or (
                            (x + 20) >= game.game_width - 20)):
                self.reward -= 20
                print('punished sackgasse')
        elif player.x_change == -20:
            if (([x, y + 20] in body) or (y + 20 >= game.game_height)) and \
                    (([x, y - 20] in body) or (y - 20 <= 0)) and \
                    (([pos[0] for pos in body[:-2] if pos[1] == y and pos[0] < x]) or ((x - 20) <= 0)):
                self.reward -= 20
                print('punished sackgasse')
        elif player.y_change == 20:
            if (([x + 20, y] in body) or (x + 20 >= game.game_width)) and \
                    (([x - 20, y] in body) or (x - 20 <= 0)) and \
                    (([pos[1] for pos in body[:-2] if pos[0] == x and pos[1] > y]) or (
                            (y + 20) >= game.game_height - 20)):
                self.reward -= 20
                print('punished sackgasse')
        elif player.y_change == -20:
            if (([x + 20, y] in body) or (x + 20 >= game.game_width)) and \
                    (([x - 20, y] in body) or (x - 20 <= 0)) and \
                    (([pos[1] for pos in body[:-2] if pos[0] == x and pos[1] < y]) or ((y - 20) <= 0)):
                self.reward -= 20
                print('punished sackgasse')


    def trace_edge(self, game, x, y, body, start_x, start_y, turn, direction, num_turns, length):
        if num_turns == 4:
            if x == start_x and y == start_y:
                return True


        if turn == 'left':
            if direction == 'upwards':
                if length > game.game_height/20 - 2:
                    return False
                if num_turns < 5 and ( [x+20, y] in body or y <= 0 ):
                    return self.trace_edge(game, x+20, y, body, start_x, start_y, 'left', 'rightwards', num_turns+1, 1)
                elif [x, y-20] in body or x <= 0:
                    return self.trace_edge(game, x, y-20, body, start_x, start_y, 'left', 'upwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'rightwards':
                if length > game.game_width/20 - 2:
                    return False
                if num_turns < 5 and ( [x, y+20] in body or x >= game.game_width ):
                    return self.trace_edge(game, x, y+20, body, start_x, start_y,  'left', 'downwards', num_turns+1, 1)
                elif [x+20, y] in body or y <= 0:
                    return self.trace_edge(game, x+20, y, body, start_x, start_y, 'left', 'rightwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'downwards':
                if length > game.game_height/20 - 2:
                    return False
                if num_turns < 5 and ( [x-20, y] in body or y >= game.game_height ):
                    return self.trace_edge(game, x-20, y, body, start_x, start_y, 'left', 'leftwards', num_turns+1, 1)
                elif [x, y+20] in body or x >= game.game_width:
                    return self.trace_edge(game, x, y+20, body, start_x, start_y, 'left', 'downwards', num_turns, length + 1)
                else:
                    return False
            else: # direction == 'leftwards'
                if length > game.game_width/20 - 2:
                    return False
                if num_turns < 5 and ( [x, y-20] in body or x <= 0 ):
                    return self.trace_edge(game, x, y-20, body, start_x, start_y, 'left', 'upwards', num_turns+1, 1)
                elif [x-20, y] in body or y >= game.game_height:
                    return self.trace_edge(game, x-20, y, body, start_x, start_y, 'left', 'leftwards', num_turns, length + 1)
                else:
                    return False
        else: # right turn
            if direction == 'upwards':
                if length > game.game_height / 20 - 2:
                    return False
                if num_turns < 5 and ( [x - 20, y] in body or y <= 0 ):
                    return self.trace_edge(game, x - 20, y, body, start_x, start_y, 'right', 'leftwards', num_turns + 1, 1)
                elif [x, y - 20] in body or x >= game.game_width:
                    return self.trace_edge(game, x, y - 20, body, start_x, start_y, 'right', 'upwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'rightwards':
                if length > game.game_width / 20 - 2:
                    return False
                if num_turns < 5 and ( [x, y - 20] in body or x >= game.game_width ):
                    return self.trace_edge(game, x, y - 20, body, start_x, start_y, 'right', 'upwards', num_turns + 1, 1)
                elif [x + 20, y] in body or y >= game.game_height:
                    return self.trace_edge(game, x + 20, y, body, start_x, start_y, 'right', 'rightwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'downwards':
                if length > game.game_height / 20 - 2:
                    return False
                if num_turns < 5 and ( [x + 20, y] in body or y >= game.game_width ):
                    return self.trace_edge(game, x + 20, y, body, start_x, start_y, 'right', 'rightwards', num_turns + 1, 1)
                elif [x, y + 20] in body or x <= 0:
                    return self.trace_edge(game, x, y + 20, body, start_x, start_y, 'right', 'downwards', num_turns, length + 1)
                else:
                    return False
            else:  # direction == 'leftwards'
                if length > game.game_width / 20 - 2:
                    return False
                if num_turns < 5 and ( [x, y + 20] in body or x <= 0 ):
                    return self.trace_edge(game, x, y + 20, body, start_x, start_y, 'right', 'downwards', num_turns + 1, 1)
                elif [x - 20, y] in body or y <= 0:
                    return self.trace_edge(game, x - 20, y, body, start_x, start_y, 'right', 'leftwards', num_turns, length + 1)
                else:
                    return False


    def punish_loop(self, game, player, curr_move):
        x = player.x
        y = player.y
        body = player.position[:-2]

        punish_value = 40
        if np.array_equal(curr_move, [0, 1, 0]): # right turn
            #TODO is player.x_change und pos schon die nach der entscheidung?
            if player.y_change == -20:
                if ([x - 20, y] in body) or (x - 20 <= 0):
                    if self.trace_edge(game, x, y + 20, body, x - 20, y, 'right', 'rightwards', 1,1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.y_change == 20:
                if ([x + 20, y] in body) or (x + 20 >= game.game_width):
                    if self.trace_edge(game, x, y - 20, body, x + 20, y, 'right', 'leftwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.x_change == -20:
                if ([x, y + 20] in body) or (y + 20 >= game.game_height):
                    if self.trace_edge(game, x + 20, y, body, x, y + 20, 'right', 'upwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            else: #elif player.x_change == 20:
                if ([x, y - 20] in body) or (y - 20 <= 0):
                    if self.trace_edge(game, x - 20, y, body, x, y - 20, 'right', 'downwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
        else: #left turn
            if player.y_change == -20:
                if ([x + 20, y] in body) or (x + 20 >= game.game_width):
                    if self.trace_edge(game, x, y + 20, body, x + 20, y, 'left', 'leftwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.y_change == 20:
                if ([x - 20, y] in body) or (x - 20 <= 0):
                    if self.trace_edge(game, x, y - 20, body, x - 20, y, 'left', 'rightwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.x_change == -20:
                if ([x, y - 20] in body) or (y - 20 <= 0):
                    if self.trace_edge(game, x + 20, y, body, x, y - 20, 'left', 'downwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            else:  # elif player.x_change == 20:
                if ([x, y + 20] in body) or (y + 20 >= game.game_width):
                    if self.trace_edge(game, x - 20, y, body, x, y + 20, 'left', 'upwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')


    def set_reward(self, game, player, crash, crash_reason, curr_move, state_old):

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
            self.reward = -0.01
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
        num_feats = Dense(50, activation='relu')(num_inp)

        code_inp = Input(shape=[self.code_length])

        x = Concatenate(axis=1)([num_feats, code_inp])

        #model.add(Dropout(0.15))
        x = Dense(80, activation='relu')(x)
        #model.add(Dropout(0.1))
        x = Dense(50, activation='relu')(x)
        #model.add(Dropout(0.05))
        output = Dense(3, activation='softmax')(x)

        model = Model([num_inp, code_inp], output)
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, board, action, reward, next_state, next_board, done):
        if len(self.memory_done) < 50000:
            #self.memory_state.append(state)
            #self.memory_action.append(action)
            #self.memory_reward.append(reward)
            #self.memory_next_state.append(next_state)
            #self.memory_done.append(done)
            #self.memory_board.append(board)
            #self.memory_next_board.append(next_board)

            self.memory.append((state, board, action, reward, next_state, next_board, done))
        else:
            #self.memory_state[self.memory_position] = state
            #self.memory_action[self.memory_position] = action
            #self.memory_reward[self.memory_position] = reward
            #self.memory_next_state[self.memory_position] = next_state
            #self.memory_done[self.memory_position] = done
            #self.memory_board.append(board)
            #self.memory_next_board.append(next_board)


            self.memory[self.memory_position] = (state, board, action, reward, next_state, next_board, done)

            self.memory_position = (self.memory_position + 1) % 50000

    def replay_new_vectorized(self):
       if len(self.memory_done) > 1000:
           rng_state = random.getstate()
           state_minibatch = np.asarray(random.sample(self.memory_state, 1000))
           random.setstate(rng_state)
           board_minibatch = np.asarray(random.sample(self.memory_board, 1000))
           random.setstate(rng_state)
           action_minibatch = np.asarray(random.sample(self.memory_action, 1000))
           random.setstate(rng_state)
           reward_minibatch = np.asarray(random.sample(self.memory_reward, 1000))
           random.setstate(rng_state)
           next_state_minibatch = np.asarray(random.sample(self.memory_next_state, 1000))
           next_board_minibatch = np.asarray(random.sample(self.memory_next_board, 1000))
           random.setstate(rng_state)
           random.setstate(rng_state)
           done_minibatch = np.asarray(random.sample(self.memory_done, 1000))
       else:
           state_minibatch = np.asarray(self.memory_state)
           board_minibatch = np.asarray(self.memory_board)
           action_minibatch = np.asarray(self.memory_action)
           reward_minibatch = np.asarray(self.memory_reward)
           next_state_minibatch = np.asarray(self.memory_next_state)
           next_board_minibatch = np.asarray(self.memory_next_board)
           done_minibatch = np.asarray(self.memory_done)

       for i in range(10):
          target = reward_minibatch[i*100:(i+1)*100]
          target[np.invert(done_minibatch[i*100:(i+1)*100])] = target[np.invert(done_minibatch[i*100:(i+1)*100])] + \
                self.gamma * np.amax(self.target_net.predict([ next_state_minibatch[i*100:(i+1)*100,:], next_board_minibatch[i*100:(i+1)*100,:] ]), 1)[np.invert(done_minibatch[i*100:(i+1)*100])]
          target_f = self.policy_net.predict([ state_minibatch[i*100:(i+1)*100,:], board_minibatch[i*100:(i+1)*100,:] ])
          target_f[:,np.argmax(action_minibatch[i*100:(i+1)*100,:],1)] = target
          self.policy_net.fit([ state_minibatch[i*100:(i+1)*100,:], board_minibatch[i*100:(i+1)*100,:] ], target_f, epochs=1, verbose=0)




    def replay_new(self):
        if len(self.memory_done) > 512:
            minibatch = random.sample(self.memory, 512)
        else:
            minibatch = self.memory

        #TODO: macht das so wirlich sinn? reward kann ja betrag von 10 haben aber die predictions sind immer normalized to norm 1
        for state, board, action, reward, next_state, next_board, done in minibatch:
           target = reward
           if not done:
               target = reward + self.gamma * np.amax(self.target_net.predict([np.array([next_state]), next_board])[0])
           target_f = self.policy_net.predict([np.array([state]), board])
           target_f[0][np.argmax(action)] = target
           self.policy_net.fit([np.array([state]), board], target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, board, action, reward, next_state, next_board, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_net.predict([next_state.reshape((1, self.state_length)), next_board])[0])
        target_f = self.policy_net.predict([state.reshape((1, self.state_length)), board])
        target_f[0][np.argmax(action)] = target
        self.policy_net.fit([state.reshape((1, self.state_length)), board], target_f, epochs=1, verbose=0)
