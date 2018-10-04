from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
import keras as ks


class Model:

    @staticmethod
    def buildModel(input_shape):
        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation=ks.activations.relu))
        model.add(Dense(128, activation=ks.activations.relu))
        model.add(Dense(15, activation=ks.activations.softmax))

        return model
