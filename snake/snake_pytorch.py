import pygame
from pygame.locals import *

from random import randint
from DQN_pytorch import DQNAgent
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from math import sqrt

# Set options to activate or deactivate the game view, and its speed
display_option = False
speed = 50
pygame.font.init()


class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('Snake')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("/Users/sigi/uni/7sem/ML/MLProject/snake/img/background.png")  # TODO: put grid in background image?
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
        self.image = pygame.image.load('/Users/sigi/uni/7sem/ML/MLProject/snake/img/snakeBody.png')
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

        if self.x < 20 or self.x > game.game_width - 40 or self.y < 20 or self.y > game.game_height - 40:
            game.crash = True
            game.crash_reason = 0
        elif [self.x, self.y] in self.position:
            game.crash = True
            game.crash_reason = 10
        eat(self, food, game)

        self.update_position(self.x, self.y)

    def display_player(self, x, y, food, game):
        self.position[-1][0] = x
        self.position[-1][1] = y

        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

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
        self.image = pygame.image.load('/Users/sigi/uni/7sem/ML/MLProject/snake/img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position and (self.x_food != player.x and self.y_food != player.y):
            return self.x_food, self.y_food
        else:
            self.food_coord(game,player)

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


def update_screen():
    pygame.display.update()
    #pygame.event.get()


def initialize_game(player, game, food, agent):
    state_init1 = agent.get_state(game, player, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = [1, 0, 0]
    player.do_move(action, player.x, player.y, game, food)
    state_init2 = agent.get_state(game, player, food)
    reward1 = agent.set_reward(game, player, game.crash, game.crash_reason, action, state_init1)

    move = np.argmax(action)
    agent.memory.push(state_init1, torch.tensor(move), state_init2, reward1)
    agent.optimize()


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
            events = pygame.event.get()
            agent.epsilon = 120 - counter_games

            if not game.human:
                state_old = agent.get_state(game, player1, food1)
                action = agent.select_action(state_old)
                final_move = np.zeros(3)
                final_move[action] = 1
            else:
                final_move = human_move(game, player1, events)
                action = np.argmax(final_move)


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
            if game.crash:
                state_new = None
            else:
                state_new = agent.get_state(game, player1, food1)

            # set treward for the new state
            reward = agent.set_reward(game, player1, game.crash, game.crash_reason, final_move, state_old)

            agent.memory.push(state_old, action, state_new, reward)


            record = get_record(game.score, record)
            if display_option: #and counter_games % 10 == 0:
                display(player1, food1, game, record)
                pygame.time.wait(speed)

        agent.optimize()
        # print(agent.epsilon)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.score)
        score_plot.append(game.score)
        counter_plot.append(counter_games)

        #if counter_games % 20 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.policy_net.save_weights('weights.hdf5')
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



run()
