from random import choice
from copy import deepcopy
import numpy as np
from math import floor

def get_immediate_danger(game, big_neighbourhood=False):
    player = game.player
    x = player.x
    y = player.y
    body = deepcopy(player.position[1:]) # compute danger for next position of body

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

    if big_neighbourhood:

        body = body[1:] # all dangers to come lie 2 steps into the future

        danger_right_straight = 0
        if player.x_change == 20 and ([x + 20, y + 20] in body or y + 20 >= game.game_height - 20  or x + 20 >= game.game_height - 20):
            danger_right_straight = 1
        elif player.x_change == -20 and ([x - 20, y - 20] in body or y - 20 < 20 or x - 20 < 20):
            danger_right_straight = 1
        elif player.y_change == 20 and ([x - 20, y + 20] in body or x - 20 < 20 or y + 20 >= game.game_width - 20):
            danger_right_straight = 1
        elif player.y_change == -20 and ([x + 20, y - 20] in body or x + 20 >= game.game_height - 20 or y - 20 < 20):
            danger_right_straight = 1

        danger_left_straight = 0
        if player.x_change == 20 and (
                [x + 20, y - 20] in body or y - 20 < 20 or x + 20 >= game.game_height - 20):
            danger_left_straight = 1
        elif player.x_change == -20 and ([x - 20, y + 20] in body or y + 20 >= game.game_width - 20 or x - 20 < 20):
            danger_left_straight = 1
        elif player.y_change == 20 and ([x + 20, y + 20] in body or x + 20 >= game.game_width - 20 or y + 20 >= game.game_width - 20):
            danger_left_straight = 1
        elif player.y_change == -20 and ([x - 20, y - 20] in body or x - 20 < 20 or y - 20 < 20):
            danger_left_straight = 1

        danger_straight_straight = 0
        if player.x_change == 20 and ([x + 40, y] in body or x + 40 >= game.game_width - 20):
            danger_straight_straight = 1
        elif player.x_change == -20 and ([x - 40, y] in body or x - 40 < 20):
            danger_straight_straight = 1
        elif player.y_change == 20 and ([x, y + 40] in body or y + 40 >= game.game_height - 20):
            danger_straight_straight = 1
        elif player.y_change == -20 and ([x, y - 40] in body or y - 40 < 20):
            danger_straight_straight = 1

        danger_right_right = 0
        if player.x_change == 20 and ([x, y + 40] in body or y + 40 >= game.game_height - 20):
            danger_right_right = 1
        elif player.x_change == -20 and ([x, y - 40] in body or y - 40 < 20):
            danger_right_right = 1
        elif player.y_change == 20 and ([x - 40, y] in body or x - 40 < 20):
            danger_right_right = 1
        elif player.y_change == -20 and ([x + 40, y] in body or x + 40 >= game.game_width - 20):
            danger_right_right = 1

        danger_left_left = 0
        if player.x_change == 20 and ([x, y - 40] in body or y - 40 < 20):
            danger_left_left = 1
        elif player.x_change == -20 and ([x, y + 40] in body or y + 40 >= game.game_height - 20):
            danger_left_left = 1
        elif player.y_change == 20 and ([x + 40, y] in body or x + 40 >= game.game_width - 20):
            danger_left_left = 1
        elif player.y_change == -20 and ([x - 40, y] in body or x - 40 < 20):
            danger_left_left = 1

        danger_right_back = 0
        if player.x_change == 20 and ([x - 20, y + 20] in body or y + 20 >= game.game_height - 20 or x - 20 < 20):
            danger_right_back = 1
        elif player.x_change == -20 and ([x + 20, y - 20] in body or y - 20 < 20 or x + 20 >= game.game_width - 20):
            danger_right_back = 1
        elif player.y_change == 20 and ([x - 20, y - 20] in body or x - 20 < 20 or y - 20 < 20):
            danger_right_back = 1
        elif player.y_change == -20 and ([x + 20, y + 20] in body or x + 20 >= game.game_width - 20 or y + 20 >= game.game_height - 20):
            danger_right_back = 1

        danger_left_back = 0
        if player.x_change == 20 and ([x - 20, y - 20] in body or y - 20 < 20 or x - 20 < 20):
            danger_left_back = 1
        elif player.x_change == -20 and ([x + 20, y + 20] in body or y + 20 >= game.game_height - 20  or x + 20 >= game.game_width - 20):
            danger_left_back = 1
        elif player.y_change == 20 and ([x + 20, y - 20] in body or x + 20 >= game.game_width - 20 or y - 20 < 20):
            danger_left_back = 1
        elif player.y_change == -20 and ([x - 20, y + 20] in body or x - 20 < 20 or y + 20 >= game.game_height - 20):
            danger_left_back = 1

        immediate_danger = [danger_straight, danger_right, danger_left, danger_right_straight, danger_left_straight, danger_straight_straight, danger_right_right, danger_left_left, danger_right_back, danger_left_back]
    else:
        immediate_danger = [danger_straight, danger_right, danger_left]
    return immediate_danger

def get_board(game, player):
    board = np.zeros([20, 20, 4])
    board[:,:,0] = 1
    if not game.crash:
        x = floor(player.x / 20) - 1
        y = floor(player.y / 20) - 1
        board[x,y,3] = 1
        board[x,y,0] = 0
        for pos in player.position[:-1]:
            x = floor(pos[0] / 20) - 1
            y = floor(pos[1] / 20) - 1
            board[x, y, 0] = 0
            board[x,y, 2] = 1
        x = floor(game.food.x_food / 20) - 1
        y = floor(game.food.y_food / 20) - 1
        if board[x,y,3] == 0:
            board[x,y,1] = 1
            board[x,y,0] = 0
    return board

def get_state(game):
    player = game.player
    food = game.food

    state = get_immediate_danger(game)

    state.append(player.x_change == -20)  # move left
    state.append(player.x_change == 20)  # move right
    state.append(player.y_change == -20)  # move up
    state.append(player.y_change == 20)  # move down
    state.append(food.x_food < player.x)  # food left
    state.append(food.x_food > player.x)  # food right
    state.append(food.y_food < player.y)  # food up
    state.append(food.y_food > player.y) # food down


    for i in range(len(state)):
        if state[i]:
            state[i]=1
        else:
            state[i]=0

    #state.append(player.consecutive_right_turns)
    #state.append(player.consecutive_left_turns)
    #state.append(player.consecutive_straight_before_turn)
    #state.append(game.player.food/game.game_width)

    #state.append(player.x/game.game_width)
    #state.append(player.y/game.game_height)
    #state.append((food.x_food - player.x) / game.game_width),  # food x difference
    #state.append((food.y_food - player.y) / game.game_height)  # food y difference

    #state.append(self.last_move[1]*self.move_count/10)
    #state.append(self.last_move[2]*self.move_count/10)

    # TODO:
    # add length of snake to state
    #state.append(player.food/game.game_width)

    # calculate distances to next wall in each direction as additional information
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
            closest = max( candidates )
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
        #player.y_change == 20:
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


    # TODO: use more rays
    # TODO: Try out distance to free space
    # TODO: lönge ja oder nein oder als kurz mittel und lange oder sowas?

    #board = self.get_board(game, player)
    #code = self.encoder.predict((np.expand_dims(board,0)))
    #board_lin = np.reshape(board, -1)

    #return np.expand_dims(np.asarray(state), 0)
    #return np.concatenate([ np.asarray(state), board_lin])
    return np.asarray(state) #, code

pixel_to_grid_transform = lambda x: (floor(x[0]/20)-1, floor(x[1]/20)-1)

def get_state_for_random_agent(game):

    danger = get_immediate_danger(game)

    for i in range(len(danger)):
        if danger[i]:
            danger[i]=1
        else:
            danger[i]=0

    return {'danger': np.asarray(danger)}

def elongate_path(G, path):
    G_full = G.copy()
    i = 0
    while i < len(path) - 1:
        G.remove_nodes_from(path)
        node = path[i]
        next_node = path[i+1]
        neighbors = list(G_full[node]._atlas.keys())
        next_neighbors = list(G_full[next_node]._atlas.keys())
        for n in neighbors:
            if G.has_node(n):
                nn = list(G[n]._atlas.keys())
                intersection =  [nod for nod in nn if nod in next_neighbors]
                if len(intersection) > 0:
                    path = path[:i+1] + [n, intersection[0]] + path[i+1:]
                    break
        i += 1

    return path

