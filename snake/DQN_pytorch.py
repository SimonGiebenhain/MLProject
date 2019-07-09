# TODO: manchmal geht schlange durch essen durch


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
from collections import namedtuple
from itertools import count
from random import randint


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    # TODO try convnet
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 130)
        self.d1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(130, 100)
        self.d2 = nn.Dropout(0.10)
        self.fc3 = nn.Linear(100,70)
        self.d3 = nn.Dropout(0.05)
        self.fc4 = nn.Linear(70, outputs)

        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.constant(self.fc1.bias, 0)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.constant(self.fc2.bias, 0)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.constant(self.fc3.bias, 0)
        nn.init.xavier_uniform(self.fc4.weight)
        nn.init.constant(self.fc4.bias, 0)
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
        #def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #    return (size - (kernel_size - 1) - 1) // stride  + 1
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        #linear_input_size = convw * convh * 32
        #self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.d1(F.relu(self.fc1(x)))
        x = self.d2(F.relu(self.fc2(x)))
        x = self.d3(F.relu(self.fc3(x)))
        return F.softmax(self.fc4(x), dim=0)

class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.GAMMA = 0.95 # TODO check whether higher values work, maybe smaller learning rate or with different representation
        self.BATCH_SIZE = 1024
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.state_length = 11
        self.learning_rate = 0.0005
        self.did_turn = 0
        self.last_move = [1, 0, 0]
        self.move_count = 0
        self.policy_net = DQN(self.state_length, 3).to(device)
        self.target_net = DQN(self.state_length, 3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #self.policy_net = self.network("weights.hdf5")
        #self.target_net = self.network("weights.hdf5")

        self.optimizer = optim.Adam(self.policy_net.parameters(), self.learning_rate)
        self.memory = ReplayMemory(50000)

        self.epsilon = 0


    def select_action(self, state):

        if self.epsilon > 0 and randint(0, 200) < self.epsilon:
            #return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

            return randint(0, 2)

        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                #action = self.policy_net(state).max(1)[1].view(1,1)
                #return action

                return np.argmax(self.policy_net(state).numpy())

    def optimize(self):
        if len(self.memory) < self.BATCH_SIZE:
            size = len(self.memory)
        else:
            size = self.BATCH_SIZE
        transitions = self.memory.sample(size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        #action_batch = torch.cat(batch.action)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        #reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    #def train_short_memory(self, state, action, reward, next_state, done):
    #    state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1))
    #    target = reward
    #    if not done:
    #        target = reward + self.gamma * np.amax(
    #            self.target_net.predict(next_state.reshape((1, self.state_length)))[0])
    #    target_f = self.policy_net.predict(state.reshape((1, self.state_length)))
    #    target_f[0][np.argmax(action)] = target
    #    self.policy_net.fit(state.reshape((1, self.state_length)), target_f, epochs=1, verbose=0)


    # TODO plot_durations()??

    def get_immediate_danger(self, game, player):
        x = player.x
        y = player.y
        body = player.position

        danger_straight = 0
        if player.x_change == 20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
            danger_straight = 1
        elif player.x_change == -20 and ([x - 20, y] in body or x - 20 < 20):
            danger_straight = 1
        elif player.y_change == 20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
            danger_straight = 1
        elif player.y_change == -20 and ([x, y - 20] in body or y - 20 < 20):
            danger_straight = 1

        danger_right = 0
        if player.x_change == 20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
            danger_right = 1
        elif player.x_change == -20 and ([x, y - 20] in body or y - 20 < 20):
            danger_right = 1
        elif player.y_change == 20 and ([x - 20, y] in body or x - 20 < 20):
            danger_right = 1
        elif player.y_change == -20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
            danger_right = 1

        danger_left = 0
        if player.x_change == 20 and ([x, y - 20] in body or y - 20 < 20):
            danger_left = 1
        elif player.x_change == -20 and ([x, y + 20] in body or y + 20 >= game.game_height - 20):
            danger_left = 1
        elif player.y_change == 20 and ([x + 20, y] in body or x + 20 >= game.game_width - 20):
            danger_left = 1
        elif player.y_change == -20 and ([x - 20, y] in body or x - 20 < 20):
            danger_left = 1

        return danger_straight, danger_right, danger_left

    # TODO: work with body position of next state instead
    def get_state(self, game, player, food):



        state = [

            (player.x_change == 20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [20, 0])) in player.position) or
                    player.position[-1][0] + 20 >= (game.game_width - 20))) or (
                    player.x_change == -20 and player.y_change == 0 and (
                    (list(map(add, player.position[-1], [-20, 0])) in player.position) or
                    player.position[-1][0] - 20 < 20)) or (
                    player.x_change == 0 and player.y_change == -20 and (
                    (list(map(add, player.position[-1], [0, -20])) in player.position) or
                    player.position[-1][-1] - 20 < 20)) or (
                    player.x_change == 0 and player.y_change == 20 and (
                    (list(map(add, player.position[-1], [0, 20])) in player.position) or
                    player.position[-1][-1] + 20 >= (game.game_height - 20))),


            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 >= (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 >= (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
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



        #state.append((food.x_food - player.x) / game.game_width),  # food x difference
        #state.append((food.y_food - player.y) / game.game_height)  # food y difference

        #state.append(self.last_move[1]*self.move_count/10)
        #state.append(self.last_move[2]*self.move_count/10)



        # TODO:
        # add length of snake to state
        #state.append(player.food/game.game_width)

        # calculate distances to next wall in each direction as additional information
 ##      if player.x_change == -20:
 #          d_wall_straight = player.position[-1][0] / game.game_width
 #          d_wall_backwards = (game.game_width - player.position[-1][0]) / game.game_width
 #          d_wall_right = player.position[-1][1] / game.game_height
 #          d_wall_left = (game.game_height - player.position[-1][1]) / game.game_height

 #      elif player.x_change == 20:
 #          d_wall_straight = (game.game_width - player.position[-1][0]) / game.game_width
 #          d_wall_backwards = player.position[-1][0] / game.game_width
 #          d_wall_right = (game.game_height - player.position[-1][1]) / game.game_height
 #          d_wall_left = player.position[-1][1] / game.game_height

 #      elif player.y_change == -20:
 #          d_wall_straight = player.position[-1][1] / game.game_height
 #          d_wall_backwards = (game.game_height - player.position[-1][1]) / game.game_height
 #          d_wall_right = (game.game_width - player.position[-1][0]) / game.game_width
 #          d_wall_left = player.position[-1][0] / game.game_width

 #      else:
 #          d_wall_straight = (game.game_height - player.position[-1][1]) / game.game_height
 #          d_wall_backwards = player.position[-1][1] / game.game_height
 #          d_wall_right = player.position[-1][0] / game.game_width
 #          d_wall_left = (game.game_width - player.position[-1][0]) / game.game_width


 #      # calculate distances to own body, if none than use distance to next wall
 #      if player.x_change == -20:
 #          x = player.position[-1][0]
 #          y = player.position[-1][1]

 #          # straight
 #          candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
 #          if candidates:
 #              closest = max( candidates )
 #          else:
 #              closest = 0
 #          d_body_straight = (x - closest) / game.game_width

 #          # right
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
 #          if candidates:
 #              closest = max(candidates)
 #          else:
 #              closest = 0
 #          d_body_right = (y - closest) / game.game_height

 #          # left
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
 #          if candidates:
 #              closest = min(candidates)
 #          else:
 #              closest = game.game_height - 20
 #          d_body_left = (closest - y) / game.game_height


 #      elif player.x_change == 20:
 #          x = player.position[-1][0]
 #          y = player.position[-1][1]

 #          # straight
 #          candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
 #          if candidates:
 #              closest = min(candidates)
 #          else:
 #              closest = game.game_width - 20
 #          d_body_straight = (closest - x) / game.game_width

 #          # right
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
 #          if candidates:
 #              closest = min(candidates)
 #          else:
 #              closest = game.game_height - 20
 #          d_body_right = (closest - y) / game.game_height

 #          # left
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
 #          if candidates:
 #              closest = max(candidates)
 #          else:
 #              closest = 0
 #          d_body_left = (y - closest) / game.game_height


 #      elif player.y_change == -20:
 #          x = player.position[-1][0]
 #          y = player.position[-1][1]

 #          # straight
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] < y]
 #          if candidates:
 #              closest = max(candidates)
 #          else:
 #              closest = 0
 #          d_body_straight = (y - closest) / game.game_height

 #          # right
 #          candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] > x]
 #          if candidates:
 #              closest = min(candidates)
 #          else:
 #              closest = game.game_width - 20
 #          d_body_right = (closest - x) / game.game_width

 #          # left
 #          candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
 #          if candidates:
 #              closest = max(candidates)
 #          else:
 #              closest = 0
 #          d_body_left = (x - closest) / game.game_width


 #      #player.y_change == 20:
 #      else:
 #          x = player.position[-1][0]
 #          y = player.position[-1][1]

 #          # straight
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
 #          if candidates:
 #              closest = min(candidates)
 #          else:
 #              closest = game.game_height - 20
 #          d_body_straight = (closest - y) / game.game_height

 #          # right
 #          candidates = [pos[0] for pos in player.position[:-2] if pos[1] == y and pos[0] < x]
 #          if candidates:
 #              closest = max(candidates)
 #          else:
 #              closest = 0
 #          d_body_right = (x - closest) / game.game_width

 #          # left
 #          candidates = [pos[1] for pos in player.position[:-2] if pos[0] == x and pos[1] > y]
 #          if candidates:
 #              closest = min(candidates)
 #          else:
 #              closest = game.game_width - 20
 #          d_body_left = (closest - x) / game.game_width

 #      state.append(d_body_straight)
 #      state.append(d_body_left)
 #      state.append(d_body_right)
 #      state.append(d_wall_right)
 #      state.append(d_wall_straight)
 #      state.append(d_wall_backwards)
 #      state.append(d_wall_left)


        # TODO: use more rays
        # TODO: Try out distance to free space
        # TODO: lönge ja oder nein oder als kurz mittel und lange oder sowas?



        return torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)


    # check whether snake is going straight into slot of width 1
    def straight_sackgasse(self, game, player):
        x = player.x
        y = player.y
        body = player.position
        if player.x_change == 20:
            if (([x, y + 20] in body) or (y + 20 >= game.game_height)) and \
                    (([x, y - 20] in body) or (y - 20 <= 0)) and \
                    (([pos[0] for pos in body[:-2] if pos[1] == y and pos[0] > x]) or (
                            (x + 20) >= game.game_width - 20)):
                self.reward -= 20
                print('punished sackgasse')
        elif player.x_change == -20:
            if (([x, y + 20] in body) or (y + 20 >= game.game_height)) and \
                    (([x, y - 20] in body) or (y - 20 <= 0)) and \
                    (([pos[0] for pos in body[:-2] if pos[1] == y and pos[0] < x]) or ((x - 20) <= 0)):
                self.reward -= 20
                print('punished sackgasse')
        elif player.y_change == 20:
            if (([x + 20, y] in body) or (x + 20 >= game.game_width)) and \
                    (([x - 20, y] in body) or (x - 20 <= 0)) and \
                    (([pos[1] for pos in body[:-2] if pos[0] == x and pos[1] > y]) or (
                            (y + 20) >= game.game_height - 20)):
                self.reward -= 20
                print('punished sackgasse')
        elif player.y_change == -20:
            if (([x + 20, y] in body) or (x + 20 >= game.game_width)) and \
                    (([x - 20, y] in body) or (x - 20 <= 0)) and \
                    (([pos[1] for pos in body[:-2] if pos[0] == x and pos[1] < y]) or ((y - 20) <= 0)):
                self.reward -= 20
                print('punished sackgasse')


    def trace_edge(self, game, x, y, body, start_x, start_y, turn, direction, num_turns, length):
        if num_turns == 4:
            if x == start_x and y == start_y:
                return True


        if turn == 'left':
            if direction == 'upwards':
                if length > game.game_height/20 - 2:
                    return False
                if num_turns < 5 and ( [x+20, y] in body or y <= 0 ):
                    return self.trace_edge(game, x+20, y, body, start_x, start_y, 'left', 'rightwards', num_turns+1, 1)
                elif [x, y-20] in body or x <= 0:
                    return self.trace_edge(game, x, y-20, body, start_x, start_y, 'left', 'upwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'rightwards':
                if length > game.game_width/20 - 2:
                    return False
                if num_turns < 5 and ( [x, y+20] in body or x >= game.game_width ):
                    return self.trace_edge(game, x, y+20, body, start_x, start_y,  'left', 'downwards', num_turns+1, 1)
                elif [x+20, y] in body or y <= 0:
                    return self.trace_edge(game, x+20, y, body, start_x, start_y, 'left', 'rightwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'downwards':
                if length > game.game_height/20 - 2:
                    return False
                if num_turns < 5 and ( [x-20, y] in body or y >= game.game_height ):
                    return self.trace_edge(game, x-20, y, body, start_x, start_y, 'left', 'leftwards', num_turns+1, 1)
                elif [x, y+20] in body or x >= game.game_width:
                    return self.trace_edge(game, x, y+20, body, start_x, start_y, 'left', 'downwards', num_turns, length + 1)
                else:
                    return False
            else: # direction == 'leftwards'
                if length > game.game_width/20 - 2:
                    return False
                if num_turns < 5 and ( [x, y-20] in body or x <= 0 ):
                    return self.trace_edge(game, x, y-20, body, start_x, start_y, 'left', 'upwards', num_turns+1, 1)
                elif [x-20, y] in body or y >= game.game_height:
                    return self.trace_edge(game, x-20, y, body, start_x, start_y, 'left', 'leftwards', num_turns, length + 1)
                else:
                    return False
        else: # right turn
            if direction == 'upwards':
                if length > game.game_height / 20 - 2:
                    return False
                if num_turns < 5 and ( [x - 20, y] in body or y <= 0 ):
                    return self.trace_edge(game, x - 20, y, body, start_x, start_y, 'right', 'leftwards', num_turns + 1, 1)
                elif [x, y - 20] in body or x >= game.game_width:
                    return self.trace_edge(game, x, y - 20, body, start_x, start_y, 'right', 'upwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'rightwards':
                if length > game.game_width / 20 - 2:
                    return False
                if num_turns < 5 and ( [x, y - 20] in body or x >= game.game_width ):
                    return self.trace_edge(game, x, y - 20, body, start_x, start_y, 'right', 'upwards', num_turns + 1, 1)
                elif [x + 20, y] in body or y >= game.game_height:
                    return self.trace_edge(game, x + 20, y, body, start_x, start_y, 'right', 'rightwards', num_turns, length + 1)
                else:
                    return False
            elif direction == 'downwards':
                if length > game.game_height / 20 - 2:
                    return False
                if num_turns < 5 and ( [x + 20, y] in body or y >= game.game_width ):
                    return self.trace_edge(game, x + 20, y, body, start_x, start_y, 'right', 'rightwards', num_turns + 1, 1)
                elif [x, y + 20] in body or x <= 0:
                    return self.trace_edge(game, x, y + 20, body, start_x, start_y, 'right', 'downwards', num_turns, length + 1)
                else:
                    return False
            else:  # direction == 'leftwards'
                if length > game.game_width / 20 - 2:
                    return False
                if num_turns < 5 and ( [x, y + 20] in body or x <= 0 ):
                    return self.trace_edge(game, x, y + 20, body, start_x, start_y, 'right', 'downwards', num_turns + 1, 1)
                elif [x - 20, y] in body or y <= 0:
                    return self.trace_edge(game, x - 20, y, body, start_x, start_y, 'right', 'leftwards', num_turns, length + 1)
                else:
                    return False


    def punish_loop(self, game, player, curr_move):
        x = player.x
        y = player.y
        body = player.position[:-2]

        punish_value = 40
        if np.array_equal(curr_move, [0, 1, 0]): # right turn
            #TODO is player.x_change und pos schon die nach der entscheidung?
            if player.y_change == -20:
                if ([x - 20, y] in body) or (x - 20 <= 0):
                    if self.trace_edge(game, x, y + 20, body, x - 20, y, 'right', 'rightwards', 1,1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.y_change == 20:
                if ([x + 20, y] in body) or (x + 20 >= game.game_width):
                    if self.trace_edge(game, x, y - 20, body, x + 20, y, 'right', 'leftwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.x_change == -20:
                if ([x, y + 20] in body) or (y + 20 >= game.game_height):
                    if self.trace_edge(game, x + 20, y, body, x, y + 20, 'right', 'upwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            else: #elif player.x_change == 20:
                if ([x, y - 20] in body) or (y - 20 <= 0):
                    if self.trace_edge(game, x - 20, y, body, x, y - 20, 'right', 'downwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
        else: #left turn
            if player.y_change == -20:
                if ([x + 20, y] in body) or (x + 20 >= game.game_width):
                    if self.trace_edge(game, x, y + 20, body, x + 20, y, 'left', 'leftwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.y_change == 20:
                if ([x - 20, y] in body) or (x - 20 <= 0):
                    if self.trace_edge(game, x, y - 20, body, x - 20, y, 'left', 'rightwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            elif player.x_change == -20:
                if ([x, y - 20] in body) or (y - 20 <= 0):
                    if self.trace_edge(game, x + 20, y, body, x, y - 20, 'left', 'downwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')
            else:  # elif player.x_change == 20:
                if ([x, y + 20] in body) or (y + 20 >= game.game_width):
                    if self.trace_edge(game, x - 20, y, body, x, y + 20, 'left', 'upwards', 1, 1):
                        self.reward -= punish_value
                        print('punished loop')


    def set_reward(self, game, player, crash, crash_reason, curr_move, state_old):

        #TODO:
        #       - sind viele kurven wirklich schlecht? immerhin kann es eine kompakte schlange geben
        #       - check for closed loops in reward and give -100 or smth.
        #       - viel platz verbrauchen bestrafen, zb. einzelne spalte oder zeile frei lassen ist nicht gut (oder ungerade anzahl)
        #           - oder halt belohnen wenn sich die schlange berührt
        #       - was noch?
        self.reward = 0
        if crash:
            #if not (state_old[0] == 1 and state_old[1] == 1 and state_old[2] == 1):
            self.reward = -10 #- crash_reason
            #else:
            #    self.reward = -5
            return self.reward
        elif player.eaten:
            self.reward = 10 #5 + player.food/10
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
        #    self.reward += 0.01
        #else:
        #    self.reward -= 0.002

        # punish going into slot of width 1
        #if player.food > 10 and not self.did_turn:
        #    self.straight_sackgasse(game, player)

        #if player.food > 10 and self.did_turn:
        #    self.punish_loop(game, player, curr_move)

        return self.reward
