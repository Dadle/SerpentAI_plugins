from serpent.game_api import GameAPI
import serpent
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras.models import load_model
from serpent.input_controller import KeyboardKey
import numpy as np
import os
import cv2
import glob
import random
import time
import scipy.misc


class KingdomAPI(GameAPI):

    def __init__(self, game=None):
        super().__init__(game=game)

    def getMoneyCount(self):
        pass

    def canBuyBuilding(self):
        pass

    def canHireBegger(self):
        return 'Maybe'

    class SpriteLocator:
        sprite_models = {}

        dropout_value = 0.2
        loss_function = 'categorical_crossentropy'  # 'mean_squared_error'
        optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False) #SGD(lr=1e-5, momentum=0.0, decay=1e-6, nesterov=False)

        wallet_positive_train_path = 'datasets\\collect_frames\\WALLET_ALL\\'
        wallet_negative_train_path = 'datasets\\collect_frames\\WALLET_MISSING\\'
        image_root_dir = "datasets\\collect_frames\\"

        #dim = (120, 77, 3)

        @classmethod
        def construct_sprite_locator_network(cls, model_name, screen_region, classes):
            """
            Defines a method to locate a specified sprite within a defined game frame
            Parameters
                model_name: name of the model to retrieve the model for use after training\n
                sprite: The name of the sprite to find as listed in game plugin sprites\n
                screen_region: geometry of the region within which to search as a tuple of window corners.
            Usage
                compiled model will be available in the dictionary sprite_models and be trainable through the train_model method of this class
            """

            # print("inside API", cls.game.window_geometry)
            input_dim = cls.util.extract_network_input_dim(screen_region, 3)
            print("Network initating with input dimemsions:", input_dim)

            model = Sequential()
            model.add(Convolution2D(filters=32, kernel_size=32, strides=(3, 3), padding='same',
                                    input_shape=input_dim,  data_format="channels_last"))
            model.add(Activation('relu'))
            model.add(Convolution2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Convolution2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Convolution2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(classes)))
            model.add(Activation('softmax'))

            # Set loss function and optimizer
            model.compile(loss=cls.loss_function, optimizer=cls.optimizer, metrics=['accuracy'])
            print("Built model for recognizing", model_name)
            print(model.summary())

            cls.sprite_models.update({model_name: {'model': model, "classes": classes}})
            print("sprite_models:", cls.sprite_models)


        @classmethod
        def archivedOldModel(cls):
            model = Sequential()

            # First CNN layer with dropout
            model.add(Convolution2D(filters=32, kernel_size=32, strides=(3, 3), input_shape=input_dim,
                                    data_format="channels_last", border_mode='same', activation='relu',
                                    W_constraint=maxnorm(3)))
            model.add(Dropout(cls.dropout_value))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
            model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())

            model.add(Dense(units=2, activation='softmax'))

            # Compile the model and add to the model list
            model.compile(loss=cls.loss_function, optimizer=cls.optimizer, metrics=['accuracy'])
            cls.sprite_models.update({model_name: model})
            print("sprite_models:", cls.sprite_models)

        @classmethod
        def sprite_recognized(cls, game_frame, screen_region, model_name, classes):
            print("Sprite recognizer got frame with shape:", game_frame.frame.shape)
            screen_region_frame = serpent.cv.extract_region_from_image(game_frame.frame, screen_region)

            scipy.misc.imsave('outfile.jpg', screen_region_frame.reshape((1,) + screen_region_frame.shape)[0])
            print("region shape:", screen_region_frame.shape, type(screen_region_frame))
            print("INPUT TO MODEL:",screen_region_frame.reshape((1,) + screen_region_frame.shape))
            prediction = cls.sprite_models[model_name]['model'].predict(screen_region_frame.reshape((1,) + screen_region_frame.shape))[0] #<----------------------------------------------------- SOMETHING HERE???!??!?
            #prediction = cls.sprite_models['WALLET']['model'].predict(game_frame.reshape((1,) + game_frame.shape))[0]
            print("Predictions made for Wallet:", prediction)
            print("Sprite recognizer have classes:", classes)
            predicted_class_index = prediction.tolist().index(max(prediction))
            print("Sprite was predicted with index:", predicted_class_index)
            return classes[predicted_class_index]



        @classmethod
        def train_model(cls, classes, model_name):
            print('Got classes:', classes)
            image_data = []
            image_class = []
            x_train = []
            y_train = []
            x_test= []
            y_test = []
            train_test_split_pct = 0.7
            #Add labeled learning examples from each folder for the sprite
            for i, name in enumerate(classes):
                files = glob.glob(os.path.join(cls.image_root_dir, name, '*.png'))
                #print("FILES:", files)
                for j, myFile in enumerate(files):
                    #print(myFile)
                    image = cv2.imread(myFile)
                    #print("image is:", np.array(image).shape)
                    image_data.append(image)
                    #print("image shape:", np.array(image_data).shape)
                    image_class.append(i)
                    if(j < train_test_split_pct * len(files)):
                        x_train.append(image)
                        y_train.append(i)
                    else:
                        x_test.append(image)
                        y_test.append(i)

            #Add negative learning examples where there is no wallet in screen region
            #files = glob.glob(os.path.join(cls.wallet_negative_train_path, '*.png'))
            #for myFile in files:
                #print(myFile)
                #image = cv2.imread(myFile)
                #image_data.append(image)
                #image_class.append(0)

            print('image_data shape:', np.array(image_data).shape, " X_label shape:", np.array(image_class).shape)

            # Make labels one-hot categorical
            #print("image_class shape:", image_class.shape)
            #print("y_train shape:", y_train.shape)
            image_class = to_categorical(image_class)
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            # Shuffle images for split into training and test datasets
            #data_class_zip = list(zip(image_data, image_class))
            #random.shuffle(data_class_zip)
            #image_data, image_class = zip(*data_class_zip)

            x_train = np.array(x_train) #image_data[:int(len(image_data)*0.7)])
            y_train = np.array(y_train) #image_class[:int(len(image_data)*0.7)])
            x_test = np.array(x_test) #image_data[int(len(image_data)*0.7):])
            y_test = np.array(y_test) #image_class[int(len(image_data)*0.7):])



            #x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            #print("img shape", train_x.shape)

            retrieved_model = cls.sprite_models[model_name]['model']
            print("got model:", retrieved_model)
            #retrieved_model.fit(np.array(image_data), np.array(image_class), epochs=100, batch_size=48, shuffle=True)
            retrieved_model.fit(x_train, y_train, epochs=100, batch_size=48, validation_data=(x_test, y_test), shuffle=True)

            # always save your weights after training or during training
            retrieved_model.save(model_name + '_trained_model.h5')

        @classmethod
        def load_model_weights(cls, model_name):
            cls.sprite_models[model_name]['model'].load_weights(model_name + '_model_weights.h5')

        @classmethod
        def load_model(cls, model_name, classes):
            cls.sprite_models.update({model_name:
                                          {'model': load_model(model_name + '_trained_model.h5'), "classes": classes}})

        @classmethod
        def generate_extra_training_examples(cls):

            datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')

            img = load_img(cls.load_dir)  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            print(type(x))

            if not os.path.exists(self.write_dir):
                os.makedirs(self.write_dir)

            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=cls.write_dir, save_prefix='WALLET', save_format='png'):
                i += 1
                if i > 20:
                    break  # otherwise the generator would loop indefinitely

        class util:

            @staticmethod
            def extract_network_input_dim(region_frame, num_channels):
                """
                :param region_frame: The corner coordinates of a screen region
                :param num_channels: The number of channels in a game_frame object during play
                :return: An input dimension tuple for a sprite locator network, eg. (y1-y0, x1-x0, channels)
                """
                print("Extracting region:", region_frame)
                return (region_frame[2] - region_frame[0], region_frame[3] - region_frame[1], num_channels)
