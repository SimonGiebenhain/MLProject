# TODO: manchmal geht schlange durch essen durch


import numpy as np
import pandas as pd
from math import floor
from random import choice
from copy import deepcopy

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

    def act(self, state):
        return choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
