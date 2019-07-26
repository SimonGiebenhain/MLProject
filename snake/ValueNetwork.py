from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Add, Activation
from keras import regularizers
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf

import random
import numpy as np
import pandas as pd
from operator import add
from math import floor

c = 1e-4

def res_block(input):
    x_skip = Conv2D(50, 3, strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(c),
                                                           bias_regularizer=regularizers.l2(c))(input)
    x = BatchNormalization()(x_skip)
    x = Activation('relu')(x)
    x = Conv2D(50, 3, strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(c),
                                                      bias_regularizer=regularizers.l2(c))(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])
    return Activation('relu')(x)

def value_network(board_dims):
    inp = Input(shape=board_dims)
    x = Conv2D(50, 3, strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(c),
                                                      bias_regularizer=regularizers.l2(c))(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)
    x = res_block(x)

    x = Conv2D(1, 1, strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x)
    output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x)

    model = Model(inp, output)

    opt = SGD(lr=0.1, momentum=0.9)
    model.compile(loss='mse', optimizer=opt)

    return model

def train():
    X = np.load('board_exp.npy')
    y = np.load('value_exp.npy')
    print(np.shape(X))
    print(np.shape(y))

    model = value_network([10,10,4])
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=20, min_lr=0.00001)
    model.fit(X, y, batch_size=32, epochs=2, shuffle=True, validation_split=0.1, callbacks=[reduce_lr])


train()