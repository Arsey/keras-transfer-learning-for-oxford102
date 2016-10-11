import os
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

import util
import config

nb_epoch = 18  # 76.07% acc
lr = 0.001
nb_train_samples, nb_validation_samples = util.get_samples_info()
classes = util.get_numbered_classes(config.nb_classes)


def save_bottleneck_features():
    model = VGG16(weights='imagenet', include_top=False)

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        config.train_data_dir,
        target_size=config.img_size,
        shuffle=False,
        classes=classes)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open(config.bf_train_path, 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        config.validation_data_dir,
        target_size=config.img_size,
        shuffle=False,
        classes=classes)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open(config.bf_valid_path, 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(config.bf_train_path, 'rb'))
    validation_data = np.load(open(config.bf_valid_path, 'rb'))

    train_labels = []
    validation_labels = []
    for i in classes:
        if i != '.DS_Store':
            i = int(i)
            train_labels += [i] * len(os.listdir('{}/{}'.format(config.train_data_dir, i)))
            validation_labels += [i] * len(os.listdir('{}/{}'.format(config.validation_data_dir, i)))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(config.output_dim, activation='relu'))
    model.add(Dense(config.output_dim, activation='relu'))
    model.add(Dense(config.nb_classes, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_data,
        train_labels,
        nb_epoch=nb_epoch,
        validation_data=(validation_data, validation_labels))

    model.save_weights(config.top_model_weights_path.format(nb_epoch, config.output_dim))

    util.save_history(history=history, prefix='bottleneck', output_dim=config.output_dim, nb_epoch=nb_epoch)


save_bottleneck_features()
train_top_model()
