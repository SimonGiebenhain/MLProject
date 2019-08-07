# TODO: manchmal geht schlange durch essen durch


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate
from keras.utils import to_categorical
from util_functions import get_immediate_danger


from keras.models import Model


import random
import numpy as np
import pandas as pd
from operator import add
from math import floor




class BiggerDQNAgent(object):

    def __init__(self):
        self.trainable = True
        self.type = "BiggerDQNAgent"
        self.reward = 0
        self.gamma = 0.95 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.agent_target = 1
        self.state_length = 25
        self.learning_rate = 0.0001
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        self.policy_net = self.network()
        self.target_net = self.network()
        self.target_net.set_weights(self.policy_net.get_weights())
        #self.policy_net = self.network("weights/BiggerDQN_8x8.hdf5")
        #self.target_net = self.network("weights/BiggerDQN_8x8.hdf5")
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


    def reset(self):
        pass

    # TODO: work with body position of next state instead
    def get_state(self, game):
        player = game.player
        food = game.food

        state = get_immediate_danger(game, True)
        danger = state[:3]

        state.append(player.x_change == -20)  # move left
        state.append(player.x_change == 20)  # move right
        state.append(player.y_change == -20)  # move up
        state.append(player.y_change == 20)  # move down
        state.append(food.x_food < player.x)  # food left
        state.append(food.x_food > player.x)  # food right
        state.append(food.y_food < player.y)  # food up
        state.append(food.y_food > player.y) # food down

        #state.append(player.consecutive_right_turns)
        #state.append(player.consecutive_left_turns)
        #state.append(player.consecutive_straight_before_turn)
        #state.append(game.player.food/game.game_width)

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

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
        #state.append(min([d_wall_right, d_body_right]))
        #state.append(min([d_wall_straight, d_body_straight]))
        state.append(d_wall_backwards)
        #state.append(min([d_wall_left, d_body_left]))
        state.append(d_wall_straight)
        state.append(d_wall_right)
        state.append(d_wall_left)

        return {'state': np.asarray(state), 'danger': danger}

    def set_reward(self, game, player, crash, steps):

        #TODO:
        #       - sind viele kurven wirklich schlecht? immerhin kann es eine kompakte schlange geben
        #       - check for closed loops in reward and give -100 or smth.
        #       - viel platz verbrauchen bestrafen, zb. einzelne spalte oder zeile frei lassen ist nicht gut (oder ungerade anzahl)
        #           - oder halt belohnen wenn sich die schlange berührt
        #       - was noch?
        self.reward = 0
        if crash:
            self.reward = -1
            #if not (state_old[0] == 1 and state_old[1] == 1 and state_old[2] == 1):
            #    self.reward = -10 #- crash_reason
            #else:
            #    self.reward = -5
            #return self.reward
        elif player.eaten:
            self.reward = 1 #5 + player.food/10
        #else:
        #    # TODO: length in state und dann simple reward funktion, vielleicht findet es selbst was gutes
        #    # TODO sonste über clean reward nachdenken
        #    #self.reward = -0.01
         #   if steps > player.food * 1.5 + 25:
         #       self.reward = - 0.05 / (player.food * 5)
        #    if player.consecutive_right_turns > 2:
        #        #if player.consecutive_straight_before_turn > 3:
        #        self.reward -= 0.1 * player.consecutive_right_turns
        #    elif player.consecutive_left_turns > 2:
        #        #if player.consecutive_straight_before_turn > 3:
        #        self.reward -= 0.1 * player.consecutive_left_turns
        #    if player.consecutive_right_turns >= 2:
        #        if player.consecutive_straight_before_turn > 1:
        #            self.reward -= 0.1 * player.consecutive_straight_before_turn
        #    elif player.consecutive_left_turns >= 2:
        #        if player.consecutive_straight_before_turn > 1:
        #            self.reward -= 0.1 * player.consecutive_straight_before_turn
        #
        #    #if player.consecutive_straight_before_turn < 2:
        #    #    self.reward -= 0.001


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

    def act(self, state):
        prediction = self.policy_net.predict(np.reshape(state['state'], [1, self.state_length]))
        final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
        return final_move

    def network(self, weights=None):

        num_inp = Input(shape=[self.state_length])
        num_feats = Dense(120, activation='relu')(num_inp)

        #model.add(Dropout(0.15))
        x = Dense(150, activation='relu')(num_feats)
        #model.add(Dropout(0.1))
        x = Dense(80, activation='relu')(x)
        #model.add(Dropout(0.05))
        output = Dense(3)(x)

        model = Model(num_inp, output)
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory_done) < 50000:
            self.memory_state.append(state)
            self.memory_action.append(action)
            self.memory_reward.append(reward)
            self.memory_next_state.append(next_state)
            self.memory_done.append(done)

            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory_state[self.memory_position] = state
            self.memory_action[self.memory_position] = action
            self.memory_reward[self.memory_position] = reward
            self.memory_next_state[self.memory_position] = next_state
            self.memory_done[self.memory_position] = done

            self.memory[self.memory_position] = (state, action, reward, next_state, done)

            self.memory_position = (self.memory_position + 1) % 50000

    def replay_new_vectorized(self):
        batch_size = 64
        if len(self.memory_done) > batch_size:
            rng_state = random.getstate()
            state_minibatch = np.squeeze(np.asarray(random.sample(self.memory_state, batch_size)))
            random.setstate(rng_state)
            action_minibatch = np.asarray(random.sample(self.memory_action, batch_size))
            random.setstate(rng_state)
            reward_minibatch = np.asarray(random.sample(self.memory_reward, batch_size))
            random.setstate(rng_state)
            next_state_minibatch = np.squeeze(np.asarray(random.sample(self.memory_next_state, batch_size)))
            random.setstate(rng_state)
            done_minibatch = np.asarray(random.sample(self.memory_done, batch_size))
        else:
            state_minibatch = np.squeeze(np.asarray(self.memory_state))
            action_minibatch = np.asarray(self.memory_action)
            reward_minibatch = np.asarray(self.memory_reward)
            next_state_minibatch = np.squeeze(np.asarray(self.memory_next_state))
            done_minibatch = np.asarray(self.memory_done)

        # for i in range(10):
        #   target = reward_minibatch[i*100:(i+1)*100]
        #   target[np.invert(done_minibatch[i*100:(i+1)*100])] = target[np.invert(done_minibatch[i*100:(i+1)*100])] + \
        #         self.gamma * np.amax(self.target_net.predict([ next_state_minibatch[i*100:(i+1)*100,:], next_board_minibatch[i*100:(i+1)*100,:] ]), 1)[np.invert(done_minibatch[i*100:(i+1)*100])]
        #   target_f = self.policy_net.predict([ state_minibatch[i*100:(i+1)*100,:], board_minibatch[i*100:(i+1)*100,:] ])
        #   target_f[:,np.argmax(action_minibatch[i*100:(i+1)*100,:],1)] = target
        #   self.policy_net.fit([ state_minibatch[i*100:(i+1)*100,:], board_minibatch[i*100:(i+1)*100,:] ], target_f, epochs=1, verbose=0)

        # for i in range(10):
        target = reward_minibatch
        target[np.invert(done_minibatch)] = target[np.invert(done_minibatch)] + self.gamma * \
                                            np.amax(self.target_net.predict(next_state_minibatch), 1)[
                                                np.invert(done_minibatch)]
        target_f = self.policy_net.predict(state_minibatch)
        target_f[:, np.argmax(action_minibatch, 1)] = target
        self.policy_net.fit(state_minibatch, target_f, epochs=1, verbose=0)

    def replay_new(self):
        if len(self.memory_done) > 128:
            minibatch = random.sample(self.memory, 128)
        else:
            minibatch = self.memory

        # TODO: macht das so wirlich sinn? reward kann ja betrag von 10 haben aber die predictions sind immer normalized to norm 1
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_net.predict(np.array([next_state]))[0])
            target_f = self.policy_net.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.policy_net.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(
                self.target_net.predict(next_state.reshape((1, self.state_length)))[0])
        target_f = self.policy_net.predict(state.reshape((1, self.state_length)))
        target_f[0][np.argmax(action)] = target
        self.policy_net.fit(state.reshape((1, self.state_length)), target_f, epochs=1, verbose=0)