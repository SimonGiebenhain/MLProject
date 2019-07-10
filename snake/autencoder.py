from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from keras import backend as K
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def construct_autoencoder():
    input_img = Input(shape=(20, 20, 2))

    code = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    code = MaxPooling2D((2, 2), padding='same')(code)
    code = Conv2D(8, (3, 3), activation='relu', padding='same')(code)
    code = MaxPooling2D((2, 2), padding='same')(code)
    code = Conv2D(4, (3, 3), activation='relu', padding='same')(code)
    code = MaxPooling2D((2, 2), padding='same')(code)

    encoder = Model(input_img, code)


    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    decoder_input = Input(shape=(3,3,4))

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    reconstruction = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(decoder_input, reconstruction)

    autoencoder_input = Input(shape=(20,20,2))
    output = decoder(encoder(autoencoder_input))
    autoencoder = Model(autoencoder_input, output)
    opt = Adam()
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    return autoencoder, encoder, decoder


def train(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train,
                    epochs=50  ,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    )

def get_data():
    x_train = np.squeeze(np.load('x_train.npy'))
    np.random.shuffle(x_train)
    #x_test = np.squeeze(np.load('x_test.npy'))
    x_test = x_train[38000:,:,:,:]
    x_train = x_train[:38000,:,:,:]
    return x_train, x_test


def visualize(board):
    img = np.zeros([20,20,3])
    img[:,:,:2] = board
    #for i in range(20):
    #    for j in range(20):
    #        if board[i,j,0] == 0:
    #            if board[i,j,1] == 1:
    #                img[i,j] = 1
    #        else:
    #            if board[i,j,1] == 0:
    #                img[i,j] = 0.66
    #            else:
    #                img[i,j] = 0.33

    plt.imshow(img)
    plt.show()



def run():
    autoencoder, encoder, decoder = construct_autoencoder()
    x_train, x_test = get_data()
    train(autoencoder, x_train, x_test)

    encoder.save_weights('encoder_weights.hdf5')
    decoder.save_weights('decoder_weights.hdf5')

def eval():
    x_train, x_test = get_data()
    autoencoder, encoder, decoder = construct_autoencoder()
    encoder.load_weights('encoder_weights.hdf5')
    decoder.load_weights('decoder_weights.hdf5')

    for i in range(20):
        test_img = x_test[80+i,:,:,:]
        test_img = test_img.astype(np.float32)
        visualize(np.squeeze(test_img))
        rest = autoencoder(np.expand_dims(test_img,0))
        visualize(rest)


run()
eval()