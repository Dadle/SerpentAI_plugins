# Keras for deep learning
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from serpent.input_controller import KeyboardKey
import numpy as np

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class KerasDeepKingdom:
    # LSTM model parameters
    dropout_value = 0.2
    activation_function = 'tanh'
    loss_function = 'categorical_crossentropy'  # 'mean_squared_error'
    optimizer = 'sgd'
    moves = {'RIGHT':KeyboardKey.KEY_RIGHT, 'LEFT': KeyboardKey.KEY_LEFT, 'DOWN':KeyboardKey.KEY_DOWN}
    moves_array = [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_LEFT, KeyboardKey.KEY_DOWN]
    move_story = []


    def __init__(self, dim):
        self.model = Sequential()

        #First recurrent layer with dropout
        self.model.add(Convolution2D(filters=1, kernel_size=32, strides=(3, 3), input_shape=dim,
                                     data_format="channels_last", border_mode='same', activation='relu',
                                     W_constraint=maxnorm(3)))
        self.model.add(Dropout(self.dropout_value))

        self.model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())

        # Recurrent part of the network
        #self.model.add(Bidirectional(LSTM(5, return_sequences=True)))
        #self.model.add(Dropout(self.dropout_value))

        #self.model.add(Bidirectional(LSTM(10, return_sequences=True)))
        #self.model.add(Dropout(self.dropout_value))


        #Output layer (returns the predicted value)
        self.model.add(Dense(units=3, activation='softmax'))

        #Set loss function and optimizer
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

    def decide(self, game_frame):
        print(np.array([game_frame]).shape)
        model_output = self.model.predict(np.array([game_frame]))
        model_likelihood_array = model_output[0].tolist()
        print("Move likelihood:", model_likelihood_array)
        max_likelihood = max(model_likelihood_array)
        move_index = model_likelihood_array.index(max_likelihood)
        print("Selected move:", self.moves_array[move_index])
        move = self.moves_array[move_index]
        self.move_story.append(move)
        return move

    def evaluate_move(self, move):
        score = 0
        if len(self.move_story) > 0:
            if move == self.move_story[len(self.move_story)-1] and move != self.moves['DOWN']:
                score = 1
            else:
                score = -1
        one_hot_score = [0, 0, 0]
        print("Move index", self.moves_array.index(move))
        one_hot_score[self.moves_array.index(move)] = score
        return one_hot_score

    def update_weights(self, game_frame, score):
        print("Got score:", score)
        self.model.fit(x=np.array([game_frame]), y=np.array([score]), batch_size=1, epochs=1,
                                          verbose=1, shuffle=False).history

