# TODO: manchmal geht schlange durch essen durch


import numpy as np
import pandas as pd
from math import floor
from random import choice
from copy import deepcopy
import networkx as nx
from util_functions import get_immediate_danger
import matplotlib.pyplot as plt




# TODO: move get state back to AGENT as method depends on agent!!!!!



pixel_to_grid_transform = lambda x: (floor(x[0]/20)-1, floor(x[1]/20)-1)



class BetterRandomAgent(object):

    def __init__(self):
        self.trainable = False
        self.reward = 0
        self.gamma = 0.95 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 25
        self.code_length = 90
        self.learning_rate = 0.0001
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        #self.policy_net = self.network("weights.hdf5")
        #self.target_net = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.type = 'BetterRandomAgent'

    # TODO implement own get_state
    def act(self, state):
        possible_actions = []
        if state[0] == 0:
            possible_actions.append([1, 0, 0])
        if state[1] == 0:
            possible_actions.append([0, 1, 0])
        if state[2] == 0:
            possible_actions.append([0, 0, 1])

        if len(possible_actions) == 0:
            return [1, 0, 0]

        return choice(possible_actions)


class SimpleRandomAgent(object):

    def __init__(self):
        self.trainable = False
        self.reward = 0
        self.gamma = 0.95 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 25
        self.code_length = 90
        self.learning_rate = 0.0001
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        #self.policy_net = self.network("weights.hdf5")
        #self.target_net = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.type = 'SimpleRandomAgent'

    # TODO implement own get_state
    # mybe link RandomAgent.get_state()
    def act(self, state):
        return choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class SimplePathAgent(object):

    def __init__(self):
        self.trainable = False
        self.state_length = 25
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        #self.policy_net = self.network("weights.hdf5")
        #self.target_net = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.type = 'SimplePathAgent'
        self.path = []
        self.path_idx = -1

    def get_state(self, game):
        player = game.player
        food = game.food
        danger = get_immediate_danger(game, big_neighbourhood=False)

        for i in range(len(danger)):
            if danger[i]:
                danger[i] = 1
            else:
                danger[i] = 0

        state = {'danger': danger, 'x_change': player.x_change, 'y_change': player.y_change, 'body': player.position,
                 'food_x': food.x_food, 'food_y': food.y_food}
        return state

    def reset(self):
        self.path = []
        self.path_idx = -1

    def act(self, state):
        x_change = int(state['x_change']/20)
        y_change = int(state['y_change']/20)
        body = state['body']
        food = (state['food_x'], state['food_y'])

        # calculate new path
        if self.path_idx == -1 or self.path_idx > len(self.path) - 2:
            G = nx.grid_2d_graph(20, 20)
            # transform from pixel to actual coordinates
            head = deepcopy(body[-1])
            head = pixel_to_grid_transform(head)
            food = pixel_to_grid_transform(food)
            body = deepcopy(body[:-1])
            body = list(map(pixel_to_grid_transform, body))

            layout = dict(zip(G,G))

            # TODO remove nodes in body
            G.remove_nodes_from(body)

            G.remove_nodes_from([(int(head[0] - x_change), int(head[1] - y_change))])

            #nx.draw(G, layout, font_size=8, node_size=50)
            #plt.show()

            #paths = nx.all_simple_paths(G, tuple(head), tuple(food))
            #self.path = next(paths)
            if not G.has_node(head):
                print('hi')
            if not G.has_node(food):
                print('hi')
            self.path = nx.shortest_path(G, head, food)
            self.path_idx = 0

        pos = self.path[self.path_idx]
        self.path_idx += 1
        next_pos = self.path[self.path_idx]
        diff_x = next_pos[0] - pos[0]
        diff_y = next_pos[1] - pos[1]
        if y_change == 1:
            if diff_y == 1:
                return [1, 0, 0]
            elif diff_x == 1:
                return [0, 0, 1]
            elif diff_x == -1:
                return [0, 1, 0]
            else:
                raise Exception('Wrong path!')
        elif y_change == -1:
            if diff_y == -1:
                return [1, 0, 0]
            elif diff_x == 1:
                return [0, 1, 0]
            elif diff_x == -1:
                return [0, 0, 1]
            else:
                raise Exception('Wrong path!')
        elif x_change == 1:
            if diff_y == 1:
                return [0, 1, 0]
            elif diff_x == 1:
                return [1, 0, 0]
            elif diff_y == -1:
                return [0, 0, 1]
            else:
                raise Exception('Wrong path!')
        else: # x_chnage == -1
            if diff_y == 1:
                return [0, 0, 1]
            elif diff_y == -1:
                return [0, 1, 0]
            elif diff_x == -1:
                return [1, 0, 0]
            else:
                raise Exception('Wrong path!')