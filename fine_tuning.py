from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

import util
import config


def tune(lr=0.0001):
    model = util.load_model(nb_class=len(config.classes), weights_path=config.get_top_model_weights_path())

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
        nb_epoch=500,
        validation_data=validation_generator,
        nb_val_samples=config.nb_validation_samples,
        callbacks=callbacks_list)

    util.save_history(history=history, prefix='fine-tuning')
