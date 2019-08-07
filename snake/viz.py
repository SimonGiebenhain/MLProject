import numpy as np
from matplotlib import pyplot as plt


def plot_avg_moves_per_apple(filname, size):
    data = np.load(filname)
    N = len(data)
    data_stage_wise = np.split(data, N/50)
    data_avg = []
    for d in data_stage_wise:
        data_avg.append(np.sum(d)/50)
    plt.plot(range(len(data_avg)),data_avg)
    plt.show()



plot_avg_moves_per_apple('log/TrainingPerformanceBiggerDQNAgent20x20.npy', 20)