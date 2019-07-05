# TODO: manchmal geht schlange durch essen durch


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 21
        self.learning_rate = 0.0005
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        self.model = self.network()
        #self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, game, player, food):

        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
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

        state.append(self.last_move[1]*self.move_count/10)
        state.append(self.last_move[2]*self.move_count/10)


        # TODO:
        # add length of snake to state
        state.append(player.food/game.game_width)

        # calculate distances to next wall in each direction as additional information
        if player.x_change == -20:
            d_wall_straight = player.position[-1][0] / game.game_width
            d_wall_backwards = 1 - d_wall_straight
            d_wall_right = player.position[-1][1] / game.game_height
            d_wall_left = 1 - d_wall_right

        elif player.x_change == 20:
            d_wall_straight = (game.game_width - player.position[-1][0]) / game.game_width
            d_wall_backwards = 1 - d_wall_straight
            d_wall_right = (game.game_height - player.position[-1][1]) / game.game_height
            d_wall_left = 1 - d_wall_right

        elif player.y_change == -20:
            d_wall_straight = player.position[-1][1] / game.game_height
            d_wall_backwards = 1 - d_wall_straight
            d_wall_right = (game.game_width - player.position[-1][1]) / game.game_width
            d_wall_left = 1 - d_wall_right

        else:
            d_wall_straight = (game.game_height - player.position[-1][1]) / game.game_height
            d_wall_backwards = 1 - d_wall_straight
            d_wall_right = player.position[-1][1] / game.game_width
            d_wall_left = 1 - d_wall_right


        # calculate distances to own body, if none than use distance to next wall
        if player.x_change == -20:
            x = player.position[-1][0]
            y = player.position[-1][1]

            candidates = [pos[x] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
            if candidates:
                closest = max( candidates )
            else:
                closest = 0
            d_body_straight = (x - closest) / game.game_width

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
            if candidates:
                closest = max(candidates)
            else:
                closest = 0
            d_body_right = (y - closest) / game.game_height

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            if candidates:
                closest = max(candidates)
            else:
                closest = game.game_height
            d_body_left = (closest - y) / game.game_height


        elif player.x_change == 20:
            x = player.position[-1][0]
            y = player.position[-1][1]

            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
            if candidates:
                closest = max(candidates)
            else:
                closest = game.game_width
            d_body_straight = (closest - x) / game.game_width

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            if candidates:
                closest = max(candidates)
            else:
                closest = game.game_height
            d_body_right = (closest - y) / game.game_height

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
            if candidates:
                closest = max(candidates)
            else:
                closest = 0
            d_body_left = (y - closest) / game.game_height


        elif player.y_change == -20:
            x = player.position[-1][0]
            y = player.position[-1][1]

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
            if candidates:
                closest = max(candidates)
            else:
                closest = 0
            d_body_straight = (x - closest) / game.game_height

            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
            if candidates:
                closest = max(candidates)
            else:
                closest = game.game_width
            d_body_right = (closest - y) / game.game_width

            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
            if candidates:
                closest = max(candidates)
            else:
                closest = 0
            d_body_left = (y - closest) / game.game_width


        #player.y_change == 20:
        else:
            x = player.position[-1][0]
            y = player.position[-1][1]

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            if candidates:
                closest = max(candidates)
            else:
                closest = game.game_height
            d_body_straight = (closest - x) / game.game_height

            candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
            if candidates:
                closest = max(candidates)
            else:
                closest = 0
            d_body_right = (y - closest) / game.game_width

            candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
            if candidates:
                closest = max(candidates)
            else:
                closest = game.game_width
            d_body_left = (closest - y) / game.game_width

        state.append(d_body_straight)
        state.append(d_body_left)
        state.append(d_body_right)
        state.append(d_wall_right)
        state.append(d_wall_straight)
        state.append(d_wall_backwards)
        state.append(d_wall_left)


        # TODO: use more rays
        # TODO: Try out distance to free space
        # TODO: lönge ja oder nein oder als kurz mittel und lange oder sowas?



        return np.asarray(state)

    def set_reward(self, player, crash, crash_reason):
        #TODO:
        #       - sind viele kurven wirklich schlecht? immerhin kann es eine kompakte schlange geben
        #       - check for closed loops in reward and give -100 or smth.
        #       - viel platz verbrauchen bestrafen, zb. einzelne spalte oder zeile frei lassen ist nicht gut (oder ungerade anzahl)
        #           - oder halt belohnen wenn sich die schlange berührt
        #       - was noch?
        self.reward = 0
        if crash:
            self.reward = -40 - crash_reason
            return self.reward
        elif player.eaten:
            self.reward = 5 + player.food/10
        #elif self.last_move == 1:
        #    self.reward += -0.03
        if player.food > 10:
            if self.move_count >= 3 and self.did_turn:
                self.reward -= self.move_count/5
                print('move count: ', self.move_count)

        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=130, activation='relu', input_dim=self.state_length))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=100, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(output_dim=70, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.state_length)))[0])
        target_f = self.model.predict(state.reshape((1, self.state_length)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.state_length)), target_f, epochs=1, verbose=0)