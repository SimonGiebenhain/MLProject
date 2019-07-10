from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.models import Model
from keras import backend as K
import numpy as np


def construct_autoencoder():
    input_img = Input(shape=(20, 20, 2))

    code = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    code = MaxPooling2D((2, 2), padding='same')(code)
    code = Conv2D(8, (3, 3), activation='relu', padding='same')(code)
    code = MaxPooling2D((2, 2), padding='same')(code)
    code = Conv2D(4, (3, 3), activation='relu', padding='same')(code)
    code = MaxPooling2D((2, 2), padding='same')(code)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    decoder_input = Input(shape=(3,3,4))

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    reconstruction = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(input_img, code)
    decoder = Model(decoder_input, reconstruction)

    autoencoder_input = Input(shape=(20,20,2))
    output = decoder(encoder(autoencoder_input))
    autoencoder = Model(autoencoder_input, output)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder, decoder


def train(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    )

def get_data():
    x_train = np.squeeze(np.load('x_train.npy'))
    #x_test = np.squeeze(np.load('x_test.npy'))
    x_test = x_train[38000:,:,:,:]
    x_train = x_train[:38000,:,:,:]
    return x_train, x_test


autoencoder, encoder, decoder = construct_autoencoder()
x_train, x_test = get_data()
train(autoencoder, x_train, x_test)

encoder.save_weights('encoder_weights.hdf5')