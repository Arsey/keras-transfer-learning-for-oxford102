import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint

import util
import config


def save_bottleneck_features():
    model = VGG16(weights='imagenet', include_top=False)

    datagen = ImageDataGenerator()
    util.apply_mean(datagen)

    generator = datagen.flow_from_directory(
        config.train_dir,
        target_size=config.img_size,
        shuffle=False,
        classes=config.classes)
    bottleneck_features_train = model.predict_generator(generator, config.nb_train_samples)
    np.save(open(config.bf_train_path, 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        config.validation_dir,
        target_size=config.img_size,
        shuffle=False,
        classes=config.classes)
    bottleneck_features_validation = model.predict_generator(generator, config.nb_validation_samples)
    np.save(open(config.bf_valid_path, 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(config.bf_train_path, 'rb'))
    validation_data = np.load(open(config.bf_valid_path, 'rb'))

    train_labels = []
    validation_labels = []
    k = 0
    for i in config.classes:
        train_labels += [k] * len(os.listdir('{}/{}'.format(config.train_dir, i)))
        validation_labels += [k] * len(os.listdir('{}/{}'.format(config.validation_dir, i)))
        k += 1

    model = util.get_top_model_for_VGG16(shape=train_data.shape[1:], nb_class=len(config.classes), W_regularizer=True)
    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

    early_stopping = EarlyStopping(verbose=1, patience=20, monitor='acc')
    model_checkpoint = ModelCheckpoint(config.top_model_weights_path, save_best_only=True, save_weights_only=True, monitor='acc')
    callbacks_list = [early_stopping, model_checkpoint]

    history = model.fit(
        train_data,
        train_labels,
        nb_epoch=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks_list)

    util.save_history(history=history, prefix='bottleneck')
