import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import random
import math
import os.path as ops

def generator(samples, batch_size=32, data_folder_path=None):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, int(batch_size / 2)):
            batch_samples = samples[offset:offset + int(batch_size / 2)]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = ops.join(data_folder_path, batch_sample[0])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                # flip the center image and angle to do data augmentation
                center_image_flip = cv2.flip(center_image, 1)
                center_angle_flip = -center_angle

                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_image_flip = cv2.cvtColor(center_image_flip, cv2.COLOR_BGR2RGB)

                images.append(center_image)
                angles.append(center_angle)
                images.append(center_image_flip)
                angles.append(center_angle_flip)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
def get_model():
    model = keras.Sequential()
    model.add(layers.Lambda((lambda x: x / 127.5 - 1.), input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
    model.add(layers.Cropping2D(cropping=((50, 20), (0, 0))))
    
    model.add(layers.Convolution2D(filters=24, kernel_size=5, strides=2, activation='relu'))
    model.add(layers.Convolution2D(filters=36, kernel_size=5, strides=2, activation='relu'))
    model.add(layers.Convolution2D(filters=48, kernel_size=5, strides=2, activation='relu'))
    model.add(layers.Convolution2D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Convolution2D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000))
    model.add(layers.Dense(100))
    model.add(layers.Dense(50))
    model.add(layers.Dense(10))
    model.add(layers.Dense(1))
    return model


def main():
    data_folder_path = 'project3_data'
    samples = []
    for data_folder in os.listdir(data_folder_path):
        if data_folder == '.DS_Store':
            continue
        elif data_folder == 'data':
            with open(data_folder_path + '/' + data_folder + '/driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for i, line in enumerate(reader):
                    if i != 0:
                        line[0] = ops.join(data_folder, 'IMG', ops.split(line[0])[-1])
                        line[1] = ops.join(data_folder, 'IMG', ops.split(line[1])[-1])
                        line[2] = ops.join(data_folder, 'IMG', ops.split(line[2])[-1])
                        samples.append(line)
                    else:
                        pass
        else:
            with open(data_folder_path + '/' + data_folder + '/driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    line[0] = ops.join(data_folder, 'IMG', ops.split(line[0])[-1])
                    line[1] = ops.join(data_folder, 'IMG', ops.split(line[1])[-1])
                    line[2] = ops.join(data_folder, 'IMG', ops.split(line[2])[-1])
                    samples.append(line)

    random.shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('samples length: {}'.format(len(samples)))
    print('train_samples length: {}'.format(len(train_samples)))
    print('validation_samples length: {}'.format(len(validation_samples)))

    # Set our batch size
    batch_size = 32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size, data_folder_path=data_folder_path)
    validation_generator = generator(validation_samples, batch_size=batch_size, data_folder_path=data_folder_path)

    model = get_model()
    model.summary()

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(2 * len(train_samples) / batch_size),
                        validation_data=validation_generator,
                        validation_steps=math.ceil(2 * len(validation_samples) / batch_size),
                        epochs=5, verbose=1)
    model.save('model.h5')


if __name__ == '__main__':
    main()
    

    