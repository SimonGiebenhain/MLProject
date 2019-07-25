import numpy as np
import pandas as pd
from math import floor
from random import choice, sample
from copy import deepcopy
import networkx as nx
from util_functions import get_immediate_danger, pixel_to_grid_transform, get_state_for_random_agent, elongate_path
import matplotlib.pyplot as plt



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

    def get_state(self, game):
        return get_state_for_random_agent(game)

    def reset(self):
        return

    # TODO implement own get_state
    # mybe link RandomAgent.get_state()
    def act(self, state):
        return choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

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

    def get_state(self, game):
        return get_state_for_random_agent(game)

    def reset(self):
        return

    # TODO implement own get_state
    def act(self, state):
        danger = state['danger']
        possible_actions = []
        if danger[0] == 0:
            possible_actions.append([1, 0, 0])
        if danger[1] == 0:
            possible_actions.append([0, 1, 0])
        if danger[2] == 0:
            possible_actions.append([0, 0, 1])

        if len(possible_actions) == 0:
            return [1, 0, 0]

        return choice(possible_actions)


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
            try:
                body.remove(head)
            except:
                pass
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
            try:
                self.path = nx.shortest_path(G, head, food)
            # If there is no path, the agent is trapped in a loop
            except:
                danger = state['danger']
                if danger[0] == 1 and danger[1] == 1and danger[2] == 1:
                    return [1, 0, 0]
                if not G.has_node(head):
                    print('hi')
                connected_component = nx.descendants(G, head)
                if len(connected_component) < 2:
                    return [1, 0, 0]
                target = sample(connected_component, 1)[0]
                if not G.has_node(head):
                    print('hi')
                if not G.has_node(target):
                    print('hi')
                self.path = nx.shortest_path(G, head, target)
                #self.path = max((path for path in nx.all_simple_paths(G, head, target)),
                #                     key=lambda path: len(path))
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

class BetterPathAgent(object):

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
        self.type = 'BetterPathAgent'
        self.path = []
        self.path_idx = -1
        self.escaping = False

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

        G_already_constrcuted = False
        # calculate new path
        if self.path_idx == -1 or self.path_idx > len(self.path) - 2:
            G_already_constrcuted = True
            G = nx.grid_2d_graph(20, 20)
            # transform from pixel to actual coordinates
            head = deepcopy(body[-1])
            head = pixel_to_grid_transform(head)
            food = pixel_to_grid_transform(food)
            body = deepcopy(body[:-1])
            body = list(map(pixel_to_grid_transform, body))

            layout = dict(zip(G,G))
            try:
                body.remove(head)
            except:
                pass
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
            try:
                self.path = nx.shortest_path(G, head, food)
                self.escaping = False
            # If there is no path, the agent is trapped in a loop
            except:
                danger = state['danger']
                if danger[0] == 1 and danger[1] == 1and danger[2] == 1:
                    return [1, 0, 0]
                if not G.has_node(head):
                    print('hi')
                connected_component = nx.descendants(G, head)
                if len(connected_component) < 2:
                    return [1, 0, 0]
                target = sample(connected_component, 1)[0]
                if not G.has_node(head):
                    print('hi')
                if not G.has_node(target):
                    print('hi')
                path = nx.shortest_path(G, head, target)
                self.path = elongate_path(G.copy(), path)
                self.escaping = True
                #self.path = max((path for path in nx.all_simple_paths(G, head, target)),
                #                     key=lambda path: len(path))
            self.path_idx = 0


        pos = self.path[self.path_idx]

        # Check whether path to fruit is free
        if self.escaping:
            if not G_already_constrcuted:
                G = nx.grid_2d_graph(20, 20)
                # transform from pixel to actual coordinates
                head = deepcopy(body[-1])
                head = pixel_to_grid_transform(head)
                food = pixel_to_grid_transform(food)
                body = deepcopy(body[:-1])
                body = list(map(pixel_to_grid_transform, body))

                layout = dict(zip(G, G))
                try:
                    body.remove(head)
                except:
                    pass
                # TODO remove nodes in body
                G.remove_nodes_from(body)

                G.remove_nodes_from([(int(head[0] - x_change), int(head[1] - y_change))])

            if nx.has_path(G, pos, food):
                self.path = nx.shortest_path(G, pos, food)
                self.path_idx = 0
                self.escaping = False
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