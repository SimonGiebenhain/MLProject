import pygame
from pygame.locals import *

from random import randint
from simple_MC_DQN import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import sqrt
from keras.utils import to_categorical


# Set options to activate or deactivate the game view, and its speed
display_option = True
speed = 50
pygame.font.init()


class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('Snake')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")  # TODO: put grid in background image?
        self.crash = False
        self.crash_reason = 0
        self.human = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0


class Player(object):

    def __init__(self, game):
        x = 0.45 * game.game_width
        y = 0.5 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = [[self.x, self.y]]
        self.food = 1
        self.food_distance = 0
        self.food_distance_old = 10  # make sure no reward for first move
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.image_head = pygame.image.load('img/snakeHead.png')
        self.x_change = 20
        self.y_change = 0
        self.consecutive_right_turns = 0
        self.consecutive_left_turns = 0
        self.consecutive_straight_before_turn = 0

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




    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(len(self.position)-1):
                x_temp, y_temp = self.position[i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))
            game.gameDisplay.blit(self.image_head, (x, y))

            update_screen()
        else:
            pygame.time.wait(5)


def correct_move(game, player, final_move):
     x = player.x
     y = player.y
     body = player.position

     danger_straight = False
     if player.x_change == 20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
        danger_straight = True
     elif player.x_change == -20 and ([x - 20, y] in body or x - 20 < 20):
        danger_straight = True
     elif player.y_change == 20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
        danger_straight = True
     elif player.y_change == -20 and ([x, y - 20] in body or y - 20 < 20):
        danger_straight = True

     danger_right = False
     if player.x_change == 20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
        danger_right = True
     elif player.x_change == -20 and ([x, y - 20] in body or y - 20 < 20):
        danger_right = True
     elif player.y_change == 20 and ([x - 20, y] in body or x - 20 < 20):
        danger_right = True
     elif player.y_change == -20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
        danger_right = True

     danger_left = False
     if player.x_change == 20 and ([x, y - 20] in body or y - 20 < 20):
        danger_left = True
     elif player.x_change == -20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
        danger_left = True
     elif player.y_change == 20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
        danger_left = True
     elif player.y_change == -20 and ([x - 20, y] in body or x - 20 < 20):
        danger_left = True

     if not (danger_straight and danger_right and danger_left):
        if np.array_equal(final_move, [1, 0, 0]) and danger_straight:
            if not danger_left:
                final_move = [0,0,1]
            else:
                final_move = [0,1,0]
        elif np.array_equal(final_move, [0, 1, 0]) and danger_right:
            if not danger_left:
                final_move = [0,0,1]
            else:
                final_move = [1,0,0]
        elif np.array_equal(final_move, [0, 0, 1]) and danger_left:
            if not danger_right:
                final_move = [0,1,0]
            else:
                final_move = [1,0,0]
     else:
        print('WTF')

     return final_move

def human_move(game, player, events):
    final_move = [1, 0, 0]
    for event in events:
        if event.type == pygame.KEYDOWN:
            if game.human and event.key == pygame.K_RIGHT:
                print('right')

                if player.y_change == 20:
                    final_move = [0, 0, 1]
                elif player.y_change == -20:
                    final_move = [0, 1, 0]
            if game.human and event.key == pygame.K_LEFT:
                print('left')

                if player.y_change == 20:
                    final_move = [0, 1, 0]
                elif player.y_change == -20:
                    final_move = [0, 0, 1]
            if game.human and event.key == pygame.K_UP:
                print('up')

                if player.x_change == 20:
                    final_move = [0, 0, 1]
                elif player.x_change == -20:
                    final_move = [0, 1, 0]
            if game.human and event.key == pygame.K_DOWN:
                print('down')

                if player.x_change == 20:
                    final_move = [0, 1, 0]
                elif player.x_change == -20:
                    final_move = [0, 0, 1]
            if event.key == pygame.K_ESCAPE:
                game.human = not game.human
    return final_move


class Food(object):

    def __init__(self):
        self.x_food = 240
        self.y_food = 200
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position and (self.x_food != player.x and self.y_food != player.y):
            return self.x_food, self.y_food
        else:
            return self.food_coord(game,player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()


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


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(player.position[-1][0], player.position[-1][1], player.food, game)
    food.display_food(food.x_food, food.y_food, game)
    pygame.event.get()


def update_screen():
    pygame.display.update()
    #pygame.event.get()


def initialize_game(player, game, food, agent):
    state_init1 = agent.get_state(game)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food)
    state_init2 = agent.get_state(game)
    reward1 = agent.set_reward(game, player, game.crash, game.crash_reason, action, state_init1, 0)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new()


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(np.array([array_counter])[0], np.array([array_score])[0], color="b", x_jitter=.1,
                     line_kws={'color': 'green'})
    ax.set(xlabel='games', ylabel='score')
    plt.show()


def run():
    pygame.init()
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
        if display_option:
            display(player1, food1, game, record)

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

            events = pygame.event.get()
            if game.human:
                final_move = human_move(game, player1, events)

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
            player1.do_move(final_move, player1.x, player1.y, game, food1)
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
            if display_option and counter_games % 50 == 0 and counter_games > 100:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

        #if counter_games > 50:
        agent.replay_new_vectorized()
        #agent.replay_new_vectorized()

        #else:
        #agent.replay_new()

        counter_games += 1
        print('Game', counter_games, '\t Score:', game.score, '\t epslion', agent.epsilon)
        score_plot.append(game.score)
        if game.score > 3:
            less_randomness += 10
        counter_plot.append(counter_games)

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

    plot_seaborn(counter_plot, score_plot)


def run_mc():
    pygame.init()
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot =[]
    record = 0
    less_randomness = 0
    old_record = 0

    eps_min = 0
    eps_max = 1
    n_games = 5000

    while counter_games < n_games:
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent)
        if display_option:
            display(player1, food1, game, record)

        action_old = [1, 0, 0]
        state_old = agent.get_state(game)
        steps = 0
        while not game.crash:
            steps += 1
            state = agent.get_state(game)
            final_move = agent.act(game, state, state_old, action_old)
            player1.do_move(final_move, player1.x, player1.y, game, food1)
            state_old = state
            action_old = final_move

            record = get_record(game.score, record)
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

        counter_games += 1
        print('Game', counter_games, '\t Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)

    plot_seaborn(counter_plot, score_plot)


def run_human():
    pygame.init()
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    less_randomness = 0
    while counter_games < 500:
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent)
        if display_option:
            display(player1, food1, game, record)

        while not game.crash:

            # TODO key listener store in final_move

            final_move = [1, 0, 0]

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        print('right')

                        if player1.y_change == 20:
                            final_move = [0, 0, 1]
                        elif player1.y_change == -20:
                            final_move = [0, 1, 0]
                    if event.key == pygame.K_LEFT:
                        print('left')

                        if player1.y_change == 20:
                            final_move = [0, 1, 0]
                        elif player1.y_change == -20:
                            final_move = [0, 0, 1]
                    if event.key == pygame.K_UP:
                        print('up')

                        if player1.x_change == 20:
                            final_move = [0, 0, 1]
                        elif player1.x_change == -20:
                            final_move = [0, 1, 0]
                    if event.key == pygame.K_DOWN:
                        print('down')

                        if player1.x_change == 20:
                            final_move = [0, 1, 0]
                        elif player1.x_change == -20:
                            final_move = [0, 0, 1]
                    if event.key == pygame.K_ESCAPE:
                        game.crash = 1

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1)

            record = get_record(game.score, record)
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    plot_seaborn(counter_plot, score_plot)


def run_agent():
    pygame.init()
    agent = DQNAgent()
    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0
    less_randomness = 0
    while counter_games < 500:
        # Initialize classes
        game = Game(440, 440)
        player1 = game.player
        food1 = game.food

        # Perform first move
        initialize_game(player1, game, food1, agent)
        if display_option:
            display(player1, food1, game, record)

        while not game.crash:

            # TODO key listener store in final_move

            final_move = [1, 0, 0]

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        print('right')

                        if player1.y_change == 20:
                            final_move = [0, 0, 1]
                        elif player1.y_change == -20:
                            final_move = [0, 1, 0]
                    if event.key == pygame.K_LEFT:
                        print('left')

                        if player1.y_change == 20:
                            final_move = [0, 1, 0]
                        elif player1.y_change == -20:
                            final_move = [0, 0, 1]
                    if event.key == pygame.K_UP:
                        print('up')

                        if player1.x_change == 20:
                            final_move = [0, 0, 1]
                        elif player1.x_change == -20:
                            final_move = [0, 1, 0]
                    if event.key == pygame.K_DOWN:
                        print('down')

                        if player1.x_change == 20:
                            final_move = [0, 1, 0]
                        elif player1.x_change == -20:
                            final_move = [0, 0, 1]
                    if event.key == pygame.K_ESCAPE:
                        game.crash = 1

            # perform new move and get new state
            player1.do_move(final_move, player1.x, player1.y, game, food1)

            record = get_record(game.score, record)
            if display_option:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)
    plot_seaborn(counter_plot, score_plot)



run_mc()

