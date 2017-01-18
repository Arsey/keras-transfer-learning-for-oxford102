from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import config
import util

# source example is here - https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes

train_top_layers_nb_epoch = 1
fine_tune_nb_epoch = 1

RELATIVITY_LAYER = 'fc1'


def create():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu', name='fc1')(x)

    # and a logistic layer
    predictions = Dense(len(config.classes), activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


def load_trained():
    model = create()
    model.load_weights(config.get_fine_tuned_weights_path())
    return model


def _get_data_generators():
    train_datagen = ImageDataGenerator(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    util.apply_mean(train_datagen)
    train_generator = train_datagen.flow_from_directory(config.train_dir, target_size=config.img_size, classes=config.classes)

    test_datagen = ImageDataGenerator()
    util.apply_mean(test_datagen)
    validation_generator = test_datagen.flow_from_directory(config.validation_dir, target_size=config.img_size, classes=config.classes)

    return train_generator, validation_generator


def _get_callbacks(weights_path, patience=20, monitor='val_acc'):
    early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
    model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
    return [early_stopping, model_checkpoint]


def train_top_layers(model):
    print("Compiling model...")
    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

    train_gen, val_gen = _get_data_generators()
    callbacks = _get_callbacks(config.get_top_model_weights_path())
    model.fit_generator(
        train_gen,
        samples_per_epoch=config.nb_train_samples,
        nb_epoch=train_top_layers_nb_epoch,
        validation_data=val_gen,
        nb_val_samples=config.nb_validation_samples,
        callbacks=callbacks)
    return model


def fine_tune_top_2_inception_blocks(model):
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    print("Compiling model...")
    model.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=["accuracy"])

    train_gen, val_gen = _get_data_generators()
    callbacks = _get_callbacks(config.get_fine_tuned_weights_path(), patience=50)

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        train_gen,
        samples_per_epoch=config.nb_train_samples,
        nb_epoch=fine_tune_nb_epoch,
        validation_data=val_gen,
        nb_val_samples=config.nb_validation_samples,
        callbacks=callbacks)


def train():
    model = create()
    model = train_top_layers(model)
    fine_tune_top_2_inception_blocks(model)
