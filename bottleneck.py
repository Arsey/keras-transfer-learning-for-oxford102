import numpy as np

np.random.seed(1337)  # for reproducibility

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint

import util
import config

util.override_keras_directory_iterator_next()

classes = util.get_classes_from_train_dir()


def save_bottleneck_features():
    model = VGG16(weights='imagenet', include_top=False)

    datagen = ImageDataGenerator()
    util.apply_mean(datagen)

    samples_info = util.get_samples_info()
    nb_train_samples = samples_info[config.train_dir]
    nb_validation_samples = samples_info[config.validation_dir]

    generator = datagen.flow_from_directory(
        config.train_dir,
        target_size=config.img_size,
        shuffle=False,
        classes=classes)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open(config.bf_train_path, 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        config.validation_dir,
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
    k = 0
    for i in classes:
        train_labels += [k] * len(os.listdir('{}/{}'.format(config.train_dir, i)))
        validation_labels += [k] * len(os.listdir('{}/{}'.format(config.validation_dir, i)))
        k += 1

    model = util.get_top_model_for_VGG16(shape=train_data.shape[1:], nb_class=len(classes), W_regularizer=True)
    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(verbose=1, patience=5)
    model_checkpoint = ModelCheckpoint(config.top_model_weights_path, save_best_only=True, save_weights_only=True)
    callbacks_list = [early_stopping, model_checkpoint]

    history = model.fit(
        train_data,
        train_labels,
        nb_epoch=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks_list)

    util.save_history(history=history, prefix='bottleneck')
