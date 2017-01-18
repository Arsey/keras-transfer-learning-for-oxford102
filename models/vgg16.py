import numpy as np
import os
import h5py as h5
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Dropout, Input

import util
import config

top_model_nb_epoch = 1
fine_tuning_nb_epoch = 1

RELATIVITY_LAYER = 'fc2'


def get_layer_weights(weights_file=None, layer_name=None):
    if not weights_file or not layer_name:
        return None
    else:
        g = weights_file[layer_name]
        weights = [g[p] for p in g]
        print 'Weights for "{}" are loaded'.format(layer_name)
        return weights


def get_top_model_for_VGG16(nb_class=None, shape=None, W_regularizer=None, weights_file_path=False, input=None, output=None):
    if not output:
        inputs = Input(shape=shape)
        x = Flatten(name='flatten')(inputs)
    else:
        x = Flatten(name='flatten', input_shape=shape)(output)

    #############################
    weights_file = None
    if weights_file_path:
        weights_file = h5.File(config.get_top_model_weights_path())

    #############################
    if W_regularizer:
        W_regularizer = l2(1e-2)

    weights_1 = get_layer_weights(weights_file, 'fc1')
    x = Dense(4096, activation='relu', W_regularizer=W_regularizer, weights=weights_1, name='fc1')(x)
    #############################

    x = Dropout(0.6)(x)

    #############################
    if W_regularizer:
        W_regularizer = l2(1e-2)

    weights_2 = get_layer_weights(weights_file, 'fc2')
    x = Dense(4096, activation='relu', W_regularizer=W_regularizer, weights=weights_2, name='fc2')(x)
    #############################

    x = Dropout(0.6)(x)

    #############################
    weights_3 = get_layer_weights(weights_file, 'predictions')
    predictions = Dense(nb_class, activation='softmax', weights=weights_3, name='predictions')(x)
    #############################

    if weights_file:
        weights_file.close()

    model = Model(input=input or inputs, output=predictions)
    return model


def load_model(nb_class, weights_path=None):
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,) + config.img_size))

    # set base_model layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in base_model.layers:
        layer.trainable = False

    model = get_top_model_for_VGG16(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        weights_file_path=weights_path,
        input=base_model.input,
        output=base_model.output)

    return model


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
        train_labels += [k] * util.get_dir_imgs_number(os.path.join(config.train_dir, i))
        validation_labels += [k] * util.get_dir_imgs_number(os.path.join(config.validation_dir, i))
        k += 1

    model = get_top_model_for_VGG16(shape=train_data.shape[1:], nb_class=len(config.classes), W_regularizer=True)
    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

    early_stopping = EarlyStopping(verbose=1, patience=20, monitor='val_acc')
    model_checkpoint = ModelCheckpoint(
        config.get_top_model_weights_path(),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_acc')
    callbacks_list = [early_stopping, model_checkpoint]

    history = model.fit(
        train_data,
        train_labels,
        nb_epoch=top_model_nb_epoch,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks_list)

    util.save_history(history=history, prefix='bottleneck')


def tune(lr=0.0001):
    model = load_model(nb_class=len(config.classes), weights_path=config.get_top_model_weights_path())

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rotation_range=30.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    util.apply_mean(train_datagen)

    train_generator = train_datagen.flow_from_directory(
        config.train_dir,
        target_size=config.img_size,
        classes=config.classes)

    test_datagen = ImageDataGenerator()
    util.apply_mean(test_datagen)

    validation_generator = test_datagen.flow_from_directory(
        config.validation_dir,
        target_size=config.img_size,
        classes=config.classes)

    early_stopping = EarlyStopping(verbose=1, patience=30, monitor='val_acc')
    model_checkpoint = ModelCheckpoint(config.get_fine_tuned_weights_path(), save_best_only=True, save_weights_only=True, monitor='val_acc')
    callbacks_list = [early_stopping, model_checkpoint]

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=config.nb_train_samples,
        nb_epoch=fine_tuning_nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=config.nb_validation_samples,
        callbacks=callbacks_list)

    util.save_history(history=history, prefix='fine-tuning')


def train():
    save_bottleneck_features()
    train_top_model()
    tune()


def load_trained():
    model = load_model(nb_class=len(config.classes))
    model.load_weights(config.get_fine_tuned_weights_path())
    return model
