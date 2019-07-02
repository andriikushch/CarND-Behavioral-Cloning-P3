import csv
from math import ceil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, BatchNormalization, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping

batch_size = 32
lines = []

# load stored data
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# helper function to read image from path
def read_image_from_disk(source_path):
    file_name = source_path.split('/')[-1]
    current_path = "./data/IMG/" + file_name
    image = cv2.imread(current_path)
    return image


# splitting data into train_samples and validation_samples
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# create a generator for memory efficiency
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images, measurements = [], []

            for sample in batch_samples:
                # create adjusted steering measurements for the center and side camera images

                # center image
                measurement = float(sample[3])
                center_image = read_image_from_disk(sample[0])
                images.append(center_image)

                measurements.append(measurement)

                images.append(cv2.flip(center_image, 1))
                measurements.append(measurement * -1.0)

                # side images
                left_image = read_image_from_disk(sample[1])
                right_image = read_image_from_disk(sample[2])

                correction = 0.2  # this is a parameter to tune
                steering_left = measurement + correction
                steering_right = measurement - correction

                measurements.extend([steering_left, steering_right])
                images.extend([left_image, right_image])

            # convert images and measurements to np.array
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                           restore_best_weights=True)]
# define model
model = Sequential()

# preprocess input normalize and crop
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

# add Convolution2D layers
model.add(Convolution2D(filters=24, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Convolution2D(filters=36, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Convolution2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))

# add fully connected layers
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples) / batch_size),
                    epochs=5, verbose=1, callbacks=callbacks)

# save result
model.save('model.h5')
