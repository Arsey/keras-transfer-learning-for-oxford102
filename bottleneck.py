import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

import util
import config

nb_epoch = 22
batch_size = 32
lr = 0.001
nb_train_samples, nb_validation_samples = util.get_samples_info()


def save_bottleneck_features():
    model = VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        config.train_data_dir,
        target_size=config.img_size,
        batch_size=batch_size,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open(config.bf_train_path, 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        config.validation_data_dir,
        target_size=config.img_size,
        batch_size=batch_size,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open(config.bf_valid_path, 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(config.bf_train_path, 'rb'))
    train_labels = []
    for i in sorted(os.listdir(config.train_data_dir)):
        if i != '.DS_Store':
            i = int(i)
            train_labels += [i] * len(os.listdir('{}/{}'.format(config.train_data_dir, i)))

    validation_data = np.load(open(config.bf_valid_path, 'rb'))
    validation_labels = []
    for i in sorted(os.listdir(config.validation_data_dir)):
        if i != '.DS_Store':
            i = int(i)
            validation_labels += [i] * len(os.listdir('{}/{}'.format(config.validation_data_dir, i)))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(config.output_dim, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(config.output_dim, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(config.nb_classes, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_data,
        train_labels,
        nb_epoch=nb_epoch,
        validation_data=(validation_data, validation_labels))

    util.save_history(
        history=history,
        prefix='bottleneck',
        lr=lr,
        output_dim=config.output_dim,
        nb_epoch=nb_epoch,
        img_size=config.img_size)

    model.save_weights(config.top_model_weights_path.format(nb_epoch, config.output_dim))


save_bottleneck_features()
train_top_model()
