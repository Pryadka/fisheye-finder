import numpy as np

import glob
from PIL import Image
import random

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, AveragePooling2D
from tensorflow.keras import Model

import keras
from keras.utils import to_categorical

from datetime import datetime
import os

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15,10]

from numpy.random import seed

seed_value = 1234567890
seed(seed_value)
tf.random.set_seed(seed_value)


NUM_CLASSES = 12
IMAGE_SIZE = 1280

# Data generator
# It takes two sorts of images, first is the background without object and second is object image with alpha channel.
# the object image receives two types of transformations: scaling and rotation. Values for these transformation are taken from random space.
# Returns array of merged images

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=32, num_classes=NUM_CLASSES):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.background_image_list = glob.glob("./data/empty/*.bmp")
        self.object_image_list = glob.glob("./data/objects/*.bmp")
        self.background_num = len(self.background_image_list)
        self.objects_num = len(self.object_image_list)

    def __len__(self):
        return int(np.floor(self.background_num / self.batch_size))

    def __getitem__(self, index):
        X, y = self.__data_generation()
        return X, y

    def on_epoch_end(self):
        pass

    def __data_generation(self):
        images = []
        labels = []
        for i in range(self.batch_size):
            background_path = random.choice(self.background_image_list)

            background = Image.open(background_path)
            label = self.num_classes - 1
            
            add_image = random.choice(['True','True', 'True','True','True','True','True','True','True','False'])

            if add_image:
                object_path = random.choice(self.object_image_list)
                image = Image.open(object_path)

                rotation_angle = random.randrange(0, 360)
                image = image.rotate(rotation_angle)
                scale_x = random.randrange(5,10)/10
                scale_y = random.randrange(5,10)/10
                image = image.resize((int(scale_x * image.size[0]), int(scale_y * image.size[1])))

                R = random.randrange(0, IMAGE_SIZE-120)
                A = random.randrange(0, 360)
                A_RAD = A / 180 * np.pi

                x = IMAGE_SIZE//2 + int(R * np.cos(A_RAD))
                y = IMAGE_SIZE//2 + int(R * np.sin(A_RAD))

                if R < 160 :
                    label = 0
                else :
                    label = int(A / (360//(self.num_classes-2))) + 1
                x_shift, y_shift = image.size[0]//2, image.size[1]//2
                background.paste(image, (x-x_shift, y-y_shift), image)
            # background = background.convert('L')
            images.append(np.asarray(background))
            labels.append(to_categorical(label, num_classes=self.num_classes))

        return np.array(images), np.array(labels)


# MODEL

NUM_FILTERS = 4
activation = 'relu'

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# like grayscale transformation with adaptive coefficients
net = Conv2D(1, kernel_size=(1,1))(inputs)

net = Conv2D(NUM_FILTERS*4, kernel_size=(3,3), activation=activation)(net)
net = MaxPooling2D(pool_size=(2,2))(net)

net = Conv2D(NUM_FILTERS*2, kernel_size=(3,3), activation=activation)(net)
net = MaxPooling2D(pool_size=(2,2))(net)

net = Conv2D(NUM_FILTERS, kernel_size=(3,3),  activation=activation)(net)
net = MaxPooling2D(pool_size=(2,2))(net)

net = Conv2D(NUM_FILTERS, kernel_size=(3,3), activation=activation)(net)
net = MaxPooling2D(pool_size=(2,2))(net)

net = Conv2D(NUM_FILTERS, kernel_size=(3,3),  activation=activation)(net)
net = MaxPooling2D(pool_size=(2,2))(net)

# performance doesn't depend on the size of this layer, let it be1
net = Conv2D(1, kernel_size=(3,3), activation=activation)(net)
net = MaxPooling2D(pool_size=(2,2))(net)

net = Flatten()(net)
outputs = Dense(NUM_CLASSES, activation='softmax')(net)

model = Model(inputs, outputs)
model.summary()

# Training
assert tf.config.list_physical_devices('GPU')
assert tf.test.is_built_with_cuda()

# optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 100
BATCH_SIZE = 8

training_generator = DataGenerator(batch_size=BATCH_SIZE)

history = model.fit(x=training_generator, epochs=EPOCHS)
# history = model.fit(train_images, labels, batch_size=BATCH_SIZE, epochs=epochs, validation_split=0.1)

def plot_history(history, path):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
    plt.legend(['Train'])
    plt.savefig(path + '/plot.png')

acc = history.history['accuracy'][-1]
print('Train Accuracy =', acc)

if acc > 0.6:
    today = datetime.now()
    result_dir_path = './results/' + today.strftime('%m%d%H%M') + '/'
    os.mkdir(result_dir_path)

    model.save(result_dir_path + '/model.keras')

    # show_history(history)
    plot_history(history, path=result_dir_path)

plt.close()