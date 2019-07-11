import random
from collections import deque

import numpy as np
import pygame
from PIL import Image
from scipy.spatial import distance
from math import floor
from random import randint

OUT_PENALTY = -1
LIFE_REWARD = 0   
APPLE_REWARD = 1


pygame.init()


class Snake:
    """
    Represents the snake and his interactions with his environment.
    """

    def __init__(self, length=3, speed=20):
        self.length = length
        self.size = int(speed)
        self.speed = int(speed)
        self.direction = None
        self.x = 0
        self.y = 0
        self.total = 0
        self.tail = None
        self.image = pygame.image.load('/Users/sigi/uni/7sem/ML/MLProject/snake/img/snakeBody.png')

    def _is_moving_backwards(self, action):
        """
        Checks if the snake is trying to move backwards (which you can't do in the game)
        :param action: The action selected by the agent
        :return: True is the action is the inverse of the snake's direction and False otherwise
        """
        # If the action selected and the direction are opposites
        if self.direction == 0 and action == 1:
            return True
        if self.direction == 1 and action == 0:
            return True
        if self.direction == 3 and action == 2:
            return True
        if self.direction == 2 and action == 3:
            return True
        else:
            return False

    def move(self, action):
        # If the snake tries to go backwards, it keeps his original direction
        if self._is_moving_backwards(action):
            action = self.direction
        else:
            self.direction = action

        if action == 0:  # LEFT
            self.x -= self.speed
        if action == 1:  # RIGHT
            self.x += self.speed
        if action == 2:  # UP
            self.y -= self.speed
        if action == 3:  # DOWN
            self.y += self.speed

        self.tail.appendleft([self.x, self.y])
        self.tail.pop()

    def eat(self):
        self.total += 1
        self.tail.appendleft([self.x, self.y])

    def dead(self, screen_width, screen_height):
        self.total = 0
        self.length = 3
        x = screen_width/2
        y = screen_height/2
        self.x = int(x - x % 20)
        self.y = int(y - y % 20)
        self.tail = deque([self.x + i * self.speed, self.y] for i in range(self.length))
        self.direction = 0

    def draw(self, screen):
        """
        Function that draws every part of the snake body.

        :param screen: pygame screen
        :param image: image that we want to draw on the screen
        """
        for i in range(len(self.tail)):
            screen.blit(self.image, (self.tail[i][0], self.tail[i][1]))


class Apple:
    """
    Represents the Apple entity, that obtains a new position when eaten.
    """

    def __init__(self):
        self.image = pygame.image.load('/Users/sigi/uni/7sem/ML/MLProject/snake/img/food2.png')
        self.x = None
        self.y = None

    def reset(self, screen_width, screen_height):
        """Resets the position of the apple at the beginning of the game."""
        x = int(screen_width/3)
        y = int(screen_height/3)
        self.x = x - x % 20
        self.y = y - y% 20

    def get_new_position(self, screen_width, screen_height, snake_tail):
        x_rand = randint(20, screen_width - 40)
        self.x = x_rand - x_rand % 20
        y_rand = randint(20, screen_height - 40)
        self.y = y_rand - y_rand % 20
        if [self.x, self.y] not in snake_tail:
            return self.x, self.y
        else:
            return self.get_new_position(screen_width, screen_height, snake_tail)

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))


class Environment:
    """
    Represents the RL environment where the agent interacts and obtains rewards associated with is actions.
    """

    screen_width = 440
    screen_height = 440

    def __init__(self, screen_width=screen_width, screen_height=screen_height):
        self.total_rewards = 0
        self._screen = pygame.display.set_mode((screen_width, screen_height + 60))
        self._screen_width = screen_width
        self._screen_height = screen_height
        self.bg = pygame.image.load("/Users/sigi/uni/7sem/ML/MLProject/snake/img/background.png")
        self._frames = None
        self._num_last_frames = 4
        self.apple = Apple()
        self.snake = Snake()
        self.record = 0

        self.reset()
        self._game_reward = 0

    def reset(self):
        if self.record < self.snake.total:
            self.record = self.snake.total
        """Reset the environment and its components."""
        self.snake.dead(self._screen_width, self._screen_height)
        self.apple.reset(self._screen_width, self._screen_height)
        self._frames = None
        self._game_reward = 0

    def get_last_frames(self, observation):
        """
        Gets the 4 previous frames of the game as the state.
        Credits goes to https://github.com/YuriyGuts/snake-ai-reinforcement.
        
        :param observation: The screenshot of the game
        :return: The state containing the 4 previous frames taken from the game
        """
        frame = observation
        if self._frames is None:
            self._frames = deque([frame] * self._num_last_frames)
        else:
            self._frames.append(frame)
            self._frames.popleft()
        state = np.concatenate(self._frames, axis=2)  # Transpose the array so the dimension of the state is (84,84,4)
        return state

    def display(self):
        self._screen.fill((255, 255, 255))

        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))
        text_score_number = myfont.render(str(self.snake.total), True, (0, 0, 0))
        text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(self.record), True, (0, 0, 0))
        self._screen.blit(text_score, (45, 440))
        self._screen.blit(text_score_number, (120, 440))
        self._screen.blit(text_highest, (190, 440))
        self._screen.blit(text_highest_number, (350, 440))
        self._screen.blit(self.bg, (10, 10))
        self.snake.draw(self._screen)
        self.apple.draw(self._screen)

        pygame.display.update()

    def screenshot(self):
        screenshot = np.zeros([20, 20, 2])
        x = floor(self.snake.x / 20) - 1
        y = floor(self.snake.y / 20) - 1
        if x < 20 and y < 20:
            screenshot[x, y, 1] = 1
        for pos in self.snake.tail:
            x = floor(pos[0] / 20) - 1
            y = floor(pos[1] / 20) - 1
            if x < 20 and y < 20:
                screenshot[x, y, 0] = 1
        x = floor(self.apple.x / 20) - 1
        y = floor(self.apple.y / 20) - 1
        if screenshot[x, y, 0] == 0:
            screenshot[x, y, 1] = 1
        return screenshot
    
    def step(self, action):
        """
        Makes the snake move according to the selected action.
        
        :param action: The action selected by the agent
        :return: The new state, the reward, and the done value
        """
        done = False
        self.snake.move(action)

        reward = LIFE_REWARD   # Reward given to stay alive

        # IF SNAKE QUITS THE SCREEEN
        if self.snake.x in [0, self.screen_width-20] or self.snake.y in [0, self.screen_height-20]:
            reward = OUT_PENALTY
            done = True


        # IF SNAKE EATS ITSELF
        head_pos = (self.snake.tail[0][0], self.snake.tail[0][1])
        for i in range(2, len(self.snake.tail)):
            body_part_pos = (self.snake.tail[i][0], self.snake.tail[i][1])
            if head_pos[0] == body_part_pos[0] and head_pos[1] == body_part_pos[1]:
                done = True
                reward = OUT_PENALTY
                break

        # IF SNAKES EATS THE APPLE
        if int(self.snake.x) == int(self.apple.x) and int(self.snake.y) == int(self.apple.y):
            self.snake.eat()
            self.apple.get_new_position(self._screen_width, self._screen_height, self.snake.tail)
            self.total_rewards += APPLE_REWARD
            self._game_reward += APPLE_REWARD
            reward = self._game_reward


        new_observation = self.screenshot()
        new_state = self.get_last_frames(new_observation)
        return new_state, reward, done
