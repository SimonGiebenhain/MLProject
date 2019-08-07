import numpy as np
import datetime
from matplotlib import pyplot as plt
import time

class PerformanceLogger(object):

    def __init__(self, num_games, agent_type, size):
        self.start_time = time.time()
        self.move_start = -1
        self.longest_move_time = -1
        self.size = str(size) + 'x' + str(size)
        self.score_history = []
        self.moves_survived = 0
        self.moves_cur_apple = 0
        self.moves_per_apple = np.zeros(size**2)
        self.n_times_at_length = np.zeros(size**2)
        self.moves_survived_history = []
        self.reason_of_death_history = []
        self.num_moves_straight = 0
        self.num_moves_right = 0
        self.num_moves_left = 0
        self.num_games = num_games
        self.agent_type = agent_type

    def log_move(self, move, cur_length, eaten):
        if self.move_start == -1:
            self.move_start = time.time()
        else:
            move_time = time.time() - self.move_start
            if move_time > self.longest_move_time:
                self.longest_move_time = move_time
        if np.array_equal(move, [1, 0, 0]):
            self.num_moves_straight += 1
        elif np.array_equal(move, [0, 1, 0]):
            self.num_moves_right += 1
        elif np.array_equal(move, [0, 0, 1]):
            self.num_moves_left += 1
        self.moves_survived += 1
        if eaten:
            if cur_length > len(self.moves_per_apple):
                cur_length = len(self.moves_per_apple) - 1
            self.moves_per_apple[cur_length] += self.moves_cur_apple
            self.n_times_at_length[cur_length] += 1
            self.moves_cur_apple = 0
        else:
            self.moves_cur_apple += 1


    def log_death(self, state, score):
        self.move_start = -1
        danger = state['danger']
        self.score_history.append(score)
        self.moves_survived_history.append(self.moves_survived)
        if danger[0] == 1 and danger[1] == 1 and danger[2] == 1:
            self.reason_of_death_history.append(1)
        else:
            self.reason_of_death_history.append(0)
        self.moves_survived = 0

    def complete_log(self, trainable):
        total_time = time.time() - self.start_time
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H:%M:%S")
        file_name = 'BA_{type}_{size}_{games}_{time}'.format(type = self.agent_type, size=self.size, games=self.num_games, time=timestamp)
        if trainable:
            file_name = 'Training_' + file_name
        file_name = './log/' + file_name
        log_file = open(file_name, 'w')
        log_file.write('Agent: \t\t {}\n'.format(self.agent_type))
        log_file.write('Number of Games: \t\t {}\n'.format(self.num_games))
        log_file.write('Avg: Score: \t\t {}\n'.format(sum(self.score_history)/self.num_games))
        log_file.write('Avg. Duration: \t\t {}\n'.format(sum(self.moves_survived_history)/self.num_games))
        log_file.write('Avg. Moves Per Apple: \t\t {}\n'.format(sum(self.moves_survived_history)/sum(self.score_history)))
        log_file.write('Avg. Reason of Death: \t\t {}\n'.format(sum(self.reason_of_death_history)/self.num_games))
        log_file.write('Best Score: \t\t {}\n'.format(max(self.score_history)))
        log_file.write('Worst Score: \t\t {}\n'.format(min(self.score_history)))
        num_moves = self.num_moves_straight + self.num_moves_right + self.num_moves_left
        log_file.write('Percentage Straight: \t\t {}\n'.format(self.num_moves_straight/num_moves))
        log_file.write('Percentage Right: \t\t {}\n'.format(self.num_moves_right/num_moves))
        log_file.write('Percentage Left: \t\t {}\n'.format(self.num_moves_left/num_moves))
        log_file.write('Avg. Move Time: \t\t {}\n'.format(total_time/num_moves))
        log_file.write('Longest Move Time: \t \t {}\n'.format(self.longest_move_time))
        log_file.close()

        #TODO plot graphs
        plt.figure()
        plt.scatter(range(len(self.score_history)), self.score_history)
        plt.xlabel('Game')
        plt.ylabel('Score')
        scatter_plot_name = file_name + '_scatter'
        plt.savefig(scatter_plot_name)
        #plt.show()

        #TODO plot distribution over scores
        plt.figure()
        bins = range(1, max(self.score_history)+2)
        plt.hist(self.score_history, bins=bins)
        plt.title('Score distribution')
        hist_name = file_name + '_hist'
        plt.savefig(hist_name)
        #plt.show()


        #TODO SAVE IMPORTANT INFORMATION FOR PLOTS WITH MULTIPLE AGENTS
        np.save('log/TrainingPerformance'+self.agent_type+self.size + '.npy', np.array(self.score_history))
        np.save('log/' + self.agent_type+ self.size +'MOVES_PER_APPLE.npy', np.array(self.moves_per_apple) / np.array(self.n_times_at_length))