from random import randint, choice
from DQN_conv2 import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import sqrt
from keras.utils import to_categorical
import gc


# Set options to activate or deactivate the game view, and its speed
display_option = False
speed = 50


class Game:

    def __init__(self, game_width, game_height):
        self.game_width = game_width
        self.game_height = game_height
        self.crash = False
        self.crash_reason = 0
        self.human = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):

    def __init__(self, game):

        self.x = 60
        self.y = 120
        self.position = [[20,120], [40, 120], [self.x, self.y]]
        self.food = 3
        self.food_distance = 0
        self.food_distance_old = 10  # make sure no reward for first move
        self.eaten = False
        self.x_change = 20
        self.y_change = 0

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

        self.food_distance_old = self.food_distance
        self.food_distance = abs(game.food.x_food - self.x) + abs(game.food.y_food - self.y)

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


        eat(self, food, game)

        if np.array_equal(move, [0,1,0]):
            if self.consecutive_left_turns > 0:
                self.consecutive_left_turns = 0
                self.consecutive_straight_before_turn = 0
            self.consecutive_right_turns += 1
        elif np.array_equal(move, [0, 0, 1]):
            if self.consecutive_right_turns > 0:
                self.consecutive_right_turns = 0
                self.consecutive_straight_before_turn = 0
            self.consecutive_left_turns += 1
        else: # straight
            self.consecutive_straight_before_turn += 1

    def do_move4(self, move, x, y, game, food):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        self.food_distance_old = self.food_distance
        self.food_distance = abs(game.food.x_food - self.x) + abs(game.food.y_food - self.y)

        if move == 0 and self.x_change != 0:
            move_array = [0, -20]
        elif move == 1 and self.y_change != 0:
            move_array = [20, 0]
        elif move == 2 and self.x_change != 0:
            move_array = [0, 20]
        elif move == 3 and self.y_change != 0:
            move_array = [-20, 0]
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

        eat(self, food, game)

class Food(object):

    def __init__(self):
        self.x_food = 100
        self.y_food = 60

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position and (self.x_food != player.x and self.y_food != player.y):
            return self.x_food, self.y_food
        else:
            return self.food_coord(game,player)


def eat(player, food, game):
    if player.x == food.x_food and player.y == food.y_food:
        food.food_coord(game, player)
        player.eaten = True
        game.score = game.score + 1


def get_record(score, record):
        if score >= record:
            return score
        else:
            return record




def initialize_game(player, game, food, agent):
    state_init1 = agent.get_state(game)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food)
    state_init2 = agent.get_state(game)
    reward1 = agent.set_reward(game, player, game.crash, game.crash_reason, action, state_init1, 0)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new()

def initialize_game_conv(player, game, food, agent):
    state_init1, board_1 = agent.get_state(game)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = 1
    player.do_move4(action, player.x, player.y, game, food)
    return board_1



def run():
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot =[]
    record = 0
    less_randomness = 0
    old_record = 0

    eps_min = 0
    eps_max = 1
    n_games = 500

    while counter_games < n_games:
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent)

        steps = 0
        while not game.crash:
            steps += 1
            if steps > 1000:
                game.crash = True
            #if step > 200 and counter_games > 80 and player1.food < 15:
            #    game.crash = True

            # agent.epsilon is set to give randomness to actions
            # agent.epsilon = 1/(2*sqrt(counter_games) + 1)
            # if counter_games > 150:
            #    agent.epsilon = 0
            agent.epsilon = max(eps_min, eps_max - (eps_max - eps_min) * 1.5*counter_games / n_games)

            # agent.epsilon = 0
            # get old state
            if not game.human:
                state_old = agent.get_state(game)

                # perform random actions based on agent.epsilon, or choose the action
                if np.random.rand() < agent.epsilon:
                    # if np.random.uniform() <= agent.epsilon:
                    final_move = to_categorical(randint(0, 2), num_classes=3)
                else:
                    # predict action based on the old state
                    #feat = np.expand_dims(state_old[:agent.state_length], 0)
                    #board = state_old[agent.state_length:]
                    #board = np.reshape(board, (1, 20, 20,1))
                    #prediction = agent.policy_net.predict([feat, board])
                    #TODO also pass board to all prediction functions
                    prediction = agent.policy_net.predict(np.reshape(state_old, [1, agent.state_length]))
                    final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
                    #final_move = correct_move(game, player1, final_move)

            else:
                final_move = [1, 0, 0]


            if np.array_equal(final_move, [1, 0, 0]):
                agent.did_turn = 0
            else:
                agent.did_turn = 1
                if np.array_equal(final_move, agent.last_move):
                    agent.move_count += 1
                else:
                    agent.last_move = final_move
                    agent.move_count = 1

            # perform new move and get new state
            player1.do_move4(final_move, player1.x, player1.y, game, food1)
            state_new = agent.get_state(game)

            if player1.eaten:
                steps = 0
            # set treward for the new state
            reward = agent.set_reward(game, player1, game.crash, game.crash_reason, final_move, state_old, steps)

            # train short memory base on the new action and state
            agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
            #agent.target_net.set_weights(agent.policy_net.get_weights())

            if not game.crash and not player1.eaten:
                if (state_old[0] == 0 and state_old[1] == 0 and state_old[2] == 0) and np.random.uniform() < 0.2:
                    agent.remember(state_old, final_move, reward, state_new, game.crash)
            else:
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            record = get_record(game.score, record)

        #if counter_games > 50:
        agent.replay_new_vectorized()
        #agent.replay_new_vectorized()

        #else:
        #agent.replay_new()

        counter_games += 1
        print('Game', counter_games, '\t Score:', game.score, '\t epslion', agent.epsilon)
        if game.score > 3:
            less_randomness += 10

        if counter_games % 10 == 0:
            agent.target_net.set_weights(agent.policy_net.get_weights())

        if old_record < record:
            old_record = record
            agent.policy_net.save_weights('vicinity10_weights.hdf5')
    #agent.policy_net.save_weights('weights.hdf5')
    #boards = np.asarray([agent.memory_board])
    #train = boards[:45000,:,:,:]
    #np.save('x_train', train)
    #test = boards[45000:,:,:,:]
    #np.save('x_test', test)


def run_conv():
    agent = DQNAgent()
    counter_games = 0
    record = 0
    old_record = 0

    eps_min = 0.2
    eps_max = 1
    n_games = 30000

    while counter_games < n_games:
        # Initialize classes
        game = Game(200, 200)
        player1 = game.player
        food1 = game.food

        # Perform first move
        board_old = initialize_game_conv(player1, game, food1, agent)


        steps_without_food = 0
        while not game.crash:
            steps_without_food += 1
            if steps_without_food > 150:
                game.crash = True

            agent.epsilon = max(eps_min, eps_max - (eps_max - eps_min) * 2*counter_games / n_games)

            # get old state
            state, board = agent.get_state(game)
            board_inp = np.concatenate([board_old, board], 2)
            # perform random actions based on agent.epsilon, or choose the action
            if np.random.rand() < agent.epsilon:
                # if np.random.uniform() <= agent.epsilon:
                if np.random.rand() < max(0.05, agent.epsilon/3):
                    final_move = randint(0,3)
                else:
                    possible_action = []
                    if state[0] == 0:
                        possible_action.append(0)
                    if state[1] == 0:
                        possible_action.append(1)
                    if state[2] == 0:
                        possible_action.append(2)
                    if state[3] == 0:
                        possible_action.append(3)
                    if len(possible_action) > 0:
                        final_move = np.array(choice(possible_action))
                    else:
                        final_move = np.array(choice([0, 1, 2, 3]))
            else:
                # predict action based on the old state
                prediction = agent.policy_net.predict([np.reshape(state, [1, agent.state_length]), np.expand_dims(board_inp,0)])
                final_move = np.argmax(prediction[0])
                #final_move = correct_move(game, player1, final_move)


            # perform new move and get new state
            player1.do_move4(final_move, player1.x, player1.y, game, food1)
            state_new, board_new = agent.get_state(game)

            if player1.eaten:
                steps_without_food = 0
            # set treward for the new state
            reward = agent.set_reward(game, player1, game.crash, game.crash_reason, final_move, state, steps_without_food)

            # train short memory base on the new action and state
            #agent.target_net.set_weights(agent.policy_net.get_weights())

            #if not game.crash and not player1.eaten:
            #    if (state[0] == 0 and state[1] == 0 and state[2] == 0) and np.random.uniform() < max(0.2, 1-agent.epsilon):
            #        agent.remember(state, board_inp, final_move, reward, state_new, np.stack([board, board_new], 2), game.crash)
            #else:
            agent.remember(state, board_inp, final_move, reward, state_new, np.concatenate([board, board_new], 2), game.crash)

            board_old = board

            record = get_record(game.score, record)

            agent.replay_new_vectorized()
            agent.target_net.set_weights(agent.policy_net.get_weights())

        counter_games += 1
        print('Game', counter_games, '\t Score:', game.score, '\t epslion', agent.epsilon)

        #todo set weights direkt nach training?
        # TODO kleinere learning rate
        # gradient clipping und huber loss
        # architektur
        # (die 4 letzten bilder) oder 3
        # yuri nachmachen
        # chance that agent doesnt kill itself1

        if old_record < record:
            old_record = record
            agent.policy_net.save_weights('conv_model_weights.hdf5')

run_conv()

