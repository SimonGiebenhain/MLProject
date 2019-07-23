import numpy as np
import datetime
from matplotlib import pyplot as plt

class PerformanceLogger(object):

    def __init__(self, num_games, agent_type):
        self.score_history = []
        self.moves_survived = 0
        self.moves_survived_history = []
        self.reason_of_death_history = []
        self.num_moves_straight = 0
        self.num_moves_right = 0
        self.num_moves_left = 0
        self.num_games = num_games
        self.agent_type = agent_type

    def log_move(self, move):
        if np.array_equal(move, [1, 0, 0]):
            self.num_moves_straight += 1
        elif np.array_equal(move, [0, 1, 0]):
            self.num_moves_right += 1
        elif np.array_equal(move, [0, 0, 1]):
            self.num_moves_left += 1
        self.moves_survived += 1


    def log_death(self, state, score):
        self.score_history.append(score)
        self.moves_survived_history.append(self.moves_survived)
        if state[0] == 1 and state[1] == 1 and state[2] == 1:
            self.reason_of_death_history.append(1)
        else:
            self.reason_of_death_history.append(0)
        self.moves_survived = 0

    def complete_log(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H:%M:%S")
        file_name = './log/{type}_{time}'.format(type = self.agent_type, time=timestamp)
        log_file = open(file_name, 'w')
        log_file.write('Agent: \t\t {}\n'.format(self.agent_type))
        log_file.write('Number of Games: \t\t {}\n'.format(self.num_games))
        log_file.write('Avg: Score: \t\t {}\n'.format(sum(self.score_history)/self.num_games))
        log_file.write('Avg. Duration: \t\t {}\n'.format(sum(self.moves_survived_history)/self.num_games))
        log_file.write('Avg. Moves Per Apple: \t\t {}\n'.format(sum(self.moves_survived_history)/sum(self.score_history)))
        log_file.write('Avg. Reason of Death: \t\t {}\n'.format(sum(self.reason_of_death_history)/self.num_games))
        log_file.write('Best Score: \t\t {}\n'.format(max(self.score_history)))
        log_file.write('Worst Score: \t\t {}\n'.format(min(self.score_history)))
        log_file.write('Percentage Straight: \t\t {}\n'.format(self.num_moves_straight/self.num_games))
        log_file.write('Percentage Right: \t\t {}\n'.format(self.num_moves_right/self.num_games))
        log_file.write('Percentage Left: \t\t {}\n'.format(self.num_moves_left/self.num_games))
        log_file.close()

        #TODO plot graphs
        plt.figure()
        plt.scatter(range(self.num_games), self.score_history)
        scatter_plot_name = file_name + '_scatter'
        plt.show()
        #plt.savefig(scatter_plot_name)

        #TODO plot distribution over scores
        plt.figure()
        bins = range(1, max(self.score_history)+2)
        plt.hist(self.score_history, bins=bins)
        plt.show()