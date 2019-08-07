from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate

from keras.models import Model
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt


import random
import numpy as np
import pandas as pd
from operator import add
from math import floor

input_space = [[0], [1]]
for k in range(10):
    input_space0 = [ el + [0] for el in input_space]
    input_space1 = [ el + [1] for el in input_space]
    input_space = input_space0 + input_space1

input_array = [np.array(el) for el in input_space]
input_array = np.stack(input_array, 0)

num_inp = Input(shape=[11])
num_feats = Dense(70, activation='relu')(num_inp)
x = Dense(80, activation='relu')(num_feats)
x = Dense(60, activation='relu')(x)
output = Dense(3)(x)

model_complete = Model(num_inp, output)
model_complete.compile(loss='mse', optimizer=Adam())
model_complete.load_weights("weights/FinalSimpleDQN20.hdf5")
#weights_list = model_complete.get_weights()

q_values = model_complete.predict(input_array)

model = Model(num_inp, x)
model.compile(loss='mse', optimizer='adam')
#model.load_weights("weights/SimpleDQNAgent.hdf5", skip_mismatch=True)
#for i, weights in enumerate(weights_list[0:9]):
#    if i > 0:
#        model.layers[i].set_weights(weights)

import matplotlib as mpl
import matplotlib.cm as cm

q_max = np.max(q_values, axis=1)
mini = np.min(q_max)
maxi = np.max(q_max)
norm = mpl.colors.Normalize(vmin=mini, vmax=maxi)
cmap = cm.hot

m = cm.ScalarMappable(norm=norm, cmap=cmap)
color_map = m.to_rgba(q_max)


representation = model.predict(input_array)
X_embedded = TSNE(n_components=2, perplexity=10).fit_transform(representation)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=color_map)
plt.show()

