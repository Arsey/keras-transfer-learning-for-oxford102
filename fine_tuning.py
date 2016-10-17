import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import util
import config

util.override_keras_directory_iterator_next()

samples_info = util.get_samples_info()
nb_train_samples = samples_info[config.train_dir]
nb_validation_samples = samples_info[config.validation_dir]


def tune(lr=0.0001):
    classes = util.get_classes_from_train_dir()

    model = util.load_model(nb_class=len(classes), weights_path=config.top_model_weights_path)

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
        classes=classes)

    test_datagen = ImageDataGenerator()
    util.apply_mean(test_datagen)

    validation_generator = test_datagen.flow_from_directory(
        config.validation_dir,
        target_size=config.img_size,
        classes=classes)

    early_stopping = EarlyStopping(verbose=1, patience=10)
    model_checkpoint = ModelCheckpoint(config.fine_tuned_weights_path, save_best_only=True, save_weights_only=True)
    callbacks_list = [early_stopping, model_checkpoint]

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=300,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=callbacks_list)

    util.save_history(history=history, prefix='fine-tuning')
