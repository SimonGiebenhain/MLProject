# TODO: manchmal geht schlange durch essen durch


from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Activation, Add
from keras.utils import to_categorical
from util_functions import get_immediate_danger
from util_functions import new_get_board as get_board

from keras.models import Model, load_model

import random
import numpy as np
import pandas as pd
from operator import add
from math import floor




class ConvDQNAgent(object):

    def __init__(self):
        self.trainable = True
        self.type = "ConvDQNAgent"
        self.reward = 0
        self.gamma = 0.99 # TODO try less aggressive value? also interval policy/target net might be too big
        self.agent_target = 1
        self.learning_rate = 0.00001
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        self.policy_net = self.network()
        #self.policy_net = load_model("weights/dqn-00024000.model")
        self.target_net = self.network()
        self.target_net.set_weights(self.policy_net.get_weights())
        self.policy_net = self.network("weights/conv_weights.hdf5")
        self.target_net = self.network("weights/conv_weights.hdf5")
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
        self.dead_memory = []
        self.dead_memory_position = 0
        self.dead_memory_state = []
        self.dead_memory_reward = []
        self.dead_memory_action = []
        self.dead_memory_next_state = []
        self.dead_memory_done = []
        self.dead_memory_board = []
        self.dead_memory_next_board = []


    def reset(self):
        pass

    # TODO: work with body position of next state instead
    def get_state(self, game):
        #danger = get_immediate_danger(game)
        state = get_board(game)
        return state



    def set_reward(self, game, player, crash, steps):

        #TODO:
        #       - sind viele kurven wirklich schlecht? immerhin kann es eine kompakte schlange geben
        #       - check for closed loops in reward and give -100 or smth.
        #       - viel platz verbrauchen bestrafen, zb. einzelne spalte oder zeile frei lassen ist nicht gut (oder ungerade anzahl)
        #           - oder halt belohnen wenn sich die schlange berührt
        #       - was noch?
        self.reward = 0
        if crash:
            danger = get_immediate_danger(game, False)
            if not(danger[0]==1 and danger[1]==1 and danger[2]==1):
                self.reward = -1

        elif player.eaten:
            self.reward = 1
        else:
            # go towards food, else get reckt
            if player.food_distance < player.food_distance_old:
                self.reward = 0.001
            else:
                self.reward = -0.001
        #    # TODO: length in state und dann simple reward funktion, vielleicht findet es selbst was gutes
        #    # TODO sonste über clean reward nachdenken
        #    #self.reward = -0.01
        #   if steps > player.food * 1.5 + 25:
        #        self.reward = - 0.05 / (player.food * 5)
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

        # punish going into slot of width 1
        #if player.food > 10 and not self.did_turn:
        #    self.straight_sackgasse(game, player)

        #if player.food > 10 and self.did_turn:
        #    self.punish_loop(game, player, curr_move)

        return self.reward

    def act(self, state):
        prediction = self.policy_net.predict(np.expand_dims(state,0))
        final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
        return final_move

    def res_block(self, x_res):
        x = Conv2D(100, 3, strides=(1, 1), padding='same')(x_res)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(100, 3, strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        return Add()([x, x_res])

    def network(self, weights=None):

        inp = Input(shape=[6,6,4])
        x = Conv2D(32, 3, strides=(1,1), padding='same')(inp)
        x = BatchNormalization()(x)
        x = (Activation('relu')(x))
        x = Conv2D(50, 3, strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = (Activation('relu')(x))
        x = Conv2D(50, 3, strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = (Activation('relu')(x))
        x = Conv2D(32, 3, strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = (Activation('relu')(x))
        x = Conv2D(10, 3, strides=(1,1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)


        #x = self.res_block(x)
        #x = self.res_block(x)
        #x = self.res_block(x)
        #x = self.res_block(x)
        #x = Activation('relu')(BatchNormalization()(Conv2D(5, 3, strides=(1,1), padding='same')(x)))
        x = (Flatten()(x))

        x = (Activation('relu')(BatchNormalization()(Dense(200)(x))))
        #x = (Activation('relu')(BatchNormalization()(Dense(100)(x))))
        #x = Dropout(rate = 0.1)(Activation('elu')(BatchNormalization()(Dense(64)(x))))
        output = Dense(3)(x)

        model = Model(inp, output)
        model.summary()
        opt = RMSprop(lr=0.00001)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory_done) < 100000:
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

            self.memory_position = (self.memory_position + 1) % 100000

    def remember_dead(self, state, action, reward, next_state, done):
        if len(self.dead_memory_done) < 50000:
            self.dead_memory_state.append(state)
            self.dead_memory_action.append(action)
            self.dead_memory_reward.append(reward)
            self.dead_memory_next_state.append(next_state)
            self.dead_memory_done.append(done)

            self.dead_memory.append((state, action, reward, next_state, done))
        else:
            self.dead_memory_state[self.dead_memory_position] = state
            self.dead_memory_action[self.dead_memory_position] = action
            self.dead_memory_reward[self.dead_memory_position] = reward
            self.dead_memory_next_state[self.dead_memory_position] = next_state
            self.dead_memory_done[self.dead_memory_position] = done

            self.dead_memory[self.dead_memory_position] = (state, action, reward, next_state, done)

            self.dead_memory_position = (self.dead_memory_position + 1) % 50000

    def replay(self):
        batch_size = min(len(self.memory_done), 64)
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

        # Reshape to match the batch structure.
        actions = np.argmax(action_minibatch, axis=1)
        rewards = reward_minibatch.repeat(3).reshape((batch_size, 3))
        episode_ends = done_minibatch.repeat(3).reshape((batch_size, 3))

        # Predict future state-action values.
        y_old = self.policy_net.predict(state_minibatch)
        y_new = self.target_net.predict(next_state_minibatch)
        Q_next = np.max(y_new, axis=1).repeat(3).reshape((batch_size, 3))

        delta = np.zeros((batch_size, 3))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y_old + delta * (rewards + self.gamma * (1 - episode_ends) * Q_next)


        dead_batch_size = min(len(self.memory_done), 64)

        rng_state = random.getstate()
        dead_state_minibatch = np.squeeze(np.asarray(random.sample(self.dead_memory_state, dead_batch_size)))
        random.setstate(rng_state)
        dead_action_minibatch = np.asarray(random.sample(self.dead_memory_action, dead_batch_size))
        random.setstate(rng_state)
        dead_reward_minibatch = np.asarray(random.sample(self.dead_memory_reward, dead_batch_size))
        random.setstate(rng_state)
        dead_next_state_minibatch = np.squeeze(np.asarray(random.sample(self.dead_memory_next_state, dead_batch_size)))
        random.setstate(rng_state)
        dead_done_minibatch = np.asarray(random.sample(self.dead_memory_done, dead_batch_size))

        # Reshape to match the batch structure.
        dead_actions = np.argmax(dead_action_minibatch, axis=1)
        dead_rewards = dead_reward_minibatch.repeat(3).reshape((dead_batch_size, 3))
        dead_episode_ends = dead_done_minibatch.repeat(3).reshape((dead_batch_size, 3))


        # Predict future state-action values.
        dead_y_old = self.policy_net.predict(dead_state_minibatch)
        dead_y_new = self.target_net.predict(dead_next_state_minibatch)
        dead_Q_next = np.max(dead_y_new, axis=1).repeat(3).reshape((dead_batch_size, 3))

        dead_delta = np.zeros((dead_batch_size, 3))
        dead_delta[np.arange(dead_batch_size), dead_actions] = 1

        dead_targets = (1 - dead_delta) * dead_y_old + dead_delta * (dead_rewards + self.gamma * (1 - dead_episode_ends) * dead_Q_next)
        self.policy_net.fit(np.concatenate([state_minibatch, dead_state_minibatch], axis=0), np.concatenate([targets, dead_targets],axis=0), epochs=1, verbose=0)

    def replay_new(self):
        batch_size = min(len(self.memory_done), 64)
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

        # Reshape to match the batch structure.
        actions = np.argmax(action_minibatch, axis=1)
        rewards = reward_minibatch.repeat(3).reshape((batch_size, 3))
        episode_ends = done_minibatch.repeat(3).reshape((batch_size, 3))

        # Predict future state-action values.
        y_old = self.policy_net.predict(state_minibatch)
        y_new = self.target_net.predict(next_state_minibatch)
        Q_next = np.max(y_new, axis=1).repeat(3).reshape((batch_size, 3))

        delta = np.zeros((batch_size, 3))
        delta[np.arange(batch_size), actions] = 1

        targets = (1 - delta) * y_old + delta * (rewards + self.gamma * (1 - episode_ends) * Q_next)

        self.policy_net.fit(state_minibatch, targets, epochs=1, verbose=0)


    def replay_dead(self):
        batch_size = min(len(self.dead_memory_done), 64)
        if batch_size > 0:
            rng_state = random.getstate()
            state_minibatch = np.squeeze(np.asarray(random.sample(self.dead_memory_state, batch_size)))
            random.setstate(rng_state)
            action_minibatch = np.asarray(random.sample(self.dead_memory_action, batch_size))
            random.setstate(rng_state)
            reward_minibatch = np.asarray(random.sample(self.dead_memory_reward, batch_size))
            random.setstate(rng_state)
            next_state_minibatch = np.squeeze(np.asarray(random.sample(self.dead_memory_next_state, batch_size)))
            random.setstate(rng_state)
            done_minibatch = np.asarray(random.sample(self.dead_memory_done, batch_size))

            # Reshape to match the batch structure.
            actions = np.argmax(action_minibatch, axis=1)
            rewards = reward_minibatch.repeat(3).reshape((batch_size, 3))
            episode_ends = done_minibatch.repeat(3).reshape((batch_size, 3))

            # Predict future state-action values.
            y_old = self.policy_net.predict(state_minibatch)
            y_new = self.target_net.predict(next_state_minibatch)
            Q_next = np.max(y_new, axis=1).repeat(3).reshape((batch_size, 3))

            delta = np.zeros((batch_size, 3))
            delta[np.arange(batch_size), actions] = 1

            targets = (1 - delta) * y_old + delta * (rewards + self.gamma * (1 - episode_ends) * Q_next)
            self.policy_net.fit(state_minibatch, targets, epochs=1, verbose=0)

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
        aamax = np.argmax(self.policy_net.predict(next_state_minibatch), 1)[np.invert(done_minibatch)]
        target[np.invert(done_minibatch)] = target[np.invert(done_minibatch)] + self.gamma * \
                                            self.target_net.predict(next_state_minibatch)[np.invert(done_minibatch), aamax]
        target_f = self.policy_net.predict(state_minibatch)
        target_f[action_minibatch.astype(bool)] = target
        self.policy_net.fit(state_minibatch, target_f, epochs=1, verbose=0)

    def replay_new_vectorized_dead(self):
        batch_size = 64
        if len(self.dead_memory_done) > batch_size:
            rng_state = random.getstate()
            state_minibatch = np.squeeze(np.asarray(random.sample(self.dead_memory_state, batch_size)))
            random.setstate(rng_state)
            action_minibatch = np.asarray(random.sample(self.dead_memory_action, batch_size))
            random.setstate(rng_state)
            reward_minibatch = np.asarray(random.sample(self.dead_memory_reward, batch_size))
            random.setstate(rng_state)
            next_state_minibatch = np.squeeze(np.asarray(random.sample(self.dead_memory_next_state, batch_size)))
            random.setstate(rng_state)
            done_minibatch = np.asarray(random.sample(self.dead_memory_done, batch_size))
            target = reward_minibatch
            aamax = np.argmax(self.policy_net.predict(next_state_minibatch), 1)[np.invert(done_minibatch)]
            target[np.invert(done_minibatch)] = target[np.invert(done_minibatch)] + self.gamma * \
                                                self.target_net.predict(next_state_minibatch)[np.invert(done_minibatch), aamax]
            target_f = self.policy_net.predict(state_minibatch)
            target_f[action_minibatch.astype(bool)] = target
            self.policy_net.fit(state_minibatch, target_f, epochs=1, verbose=0)

    #def replay_new(self):
    #   if len(self.memory_done) > 128:
    #       minibatch = random.sample(self.memory, 128)
    #   else:
    #       minibatch = self.memory

    #   # TODO: macht das so wirlich sinn? reward kann ja betrag von 10 haben aber die predictions sind immer normalized to norm 1
    #   for state, action, reward, next_state, done in minibatch:
    #       target = reward
    #       if not done:
    #           target = reward + self.gamma * np.amax(self.target_net.predict(np.array([next_state]))[0])
    #       target_f = self.policy_net.predict(np.array([state]))
    #       target_f[0][np.argmax(action)] = target
    #       self.policy_net.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(
                self.policy_net.predict(np.expand_dims(next_state, 0))[0])
        target_f = self.policy_net.predict(np.expand_dims(state, 0))
        target_f[0][np.argmax(action)] = target
        self.policy_net.fit(np.expand_dims(state, 0), target_f, epochs=1, verbose=0)