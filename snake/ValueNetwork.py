from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers import Input, Conv2D, BatchNormalization, Concatenate, Add, Activation
from keras import regularizers
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from keras.losses import mse, categorical_crossentropy

import tensorflow as tf

import random
import numpy as np
import pandas as pd
from operator import add
from math import floor
from util_functions import shuffle_in_unison

INIT_LR = 0.01
EPOCHS = 5
BATCH_SIZE = 32

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

    #opt = SGD(lr=0.1, momentum=0.9)
    opt = Adam()
    model.compile(loss='mse', optimizer=opt)

    return model


def combined_network(board_dims):
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

    x_value = Conv2D(1, 1, strides=(1,1), kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x)
    x_value = BatchNormalization()(x_value)
    x_value = Activation('relu')(x_value)
    x_value = Flatten()(x_value)

    x_value = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x_value)
    value = Dense(1, activation='sigmoid', name='value_output',
                  kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x_value)


    x_policy = Conv2D(2, 1, strides=(1,1), kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x)
    x_policy = BatchNormalization()(x_policy)
    x_policy = Activation('relu')(x_policy)
    x_policy = Flatten()(x_policy)
    policy = Dense(3, activation='softmax', name='policy_output',
             kernel_regularizer=regularizers.l2(c), bias_regularizer=regularizers.l2(c))(x_policy)

    model = Model(inp, [value, policy])

    losses = {
        "value_output": "mse",
        "policy_output": "categorical_crossentropy",
    }
    lossWeights = {"value_output": 0.01, "policy_output": 1.0}

    # initialize the optimizer and compile the model
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights)

    return model

def train():
    X = np.load('board_exp.npy')
    y_value = np.load('value_exp.npy')
    y_policy = np.load('move_exp.npy')

    X, y_value, y_policy = shuffle_in_unison(X, y_value, y_policy)

    model = combined_network([10,10,4])
    model.summary()

    #TODO ceckpoint callback
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                              patience=5, min_lr=0.000001, verbose=1)
    model.fit(X,
              {'value_output': y_value, 'policy_output': y_policy},
              batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=0.1)#, callbacks=[reduce_lr])

train()