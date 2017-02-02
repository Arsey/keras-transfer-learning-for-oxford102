from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import defaultdict
import os
import scipy.misc
import numpy as np
import config
import util
from keras.utils import np_utils
from keras.preprocessing import image

# source example is here - https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes

train_top_layers_nb_epoch = 100
fine_tune_nb_epoch = 200

RELATIVITY_LAYER = 'fc1'


def preprocess_input(x0):
    x = x0 / 255.
    x -= 0.5
    x *= 2.
    return x


def reverse_preprocess_input(x0):
    x = x0 / 2.0
    x += 0.5
    x *= 255.
    return x


def dataset(base_dir, n):
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)

    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
            img = scipy.misc.imread(filename, mode='RGB')
            height, width, chan = img.shape
            assert chan == 3
            aspect_ratio = float(max((height, width))) / min((height, width))
            if aspect_ratio > 2:
                continue
            # We pick the largest center square.
            centery = height // 2
            centerx = width // 2
            radius = min((centerx, centery))
            img = img[centery - radius:centery + radius, centerx - radius:centerx + radius]
            img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
            X.append(img)
            y.append(class_index)
            useful_image_count += 1
    print "processed %d, used %d" % (processed_image_count, useful_image_count)

    X = np.array(X).astype(np.float32)
    X = X.transpose((0, 3, 1, 2))
    X = preprocess_input(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print "classes:"
    for class_index, class_name in enumerate(tags):
        print class_name, sum(y == class_index)
    print

    return X, y, tags


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


def _get_data_generators(train_datagen):
    util.apply_mean(train_datagen)
    train_generator = train_datagen.flow_from_directory(config.train_dir, target_size=config.img_size, classes=config.classes,
                                                        shuffle=False
                                                        )

    test_datagen = ImageDataGenerator()
    util.apply_mean(test_datagen)
    validation_generator = test_datagen.flow_from_directory(config.validation_dir, target_size=config.img_size, classes=config.classes,
                                                            shuffle=False
                                                            )

    return train_generator, validation_generator


def _get_callbacks(weights_path, patience=20, monitor='val_loss'):
    early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
    model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
    return [early_stopping, model_checkpoint]


def train_top_layers(model, X_train, Y_train, X_test, Y_test, datagen):
    print("Compiling model...")
    # rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train_datagen = ImageDataGenerator(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # train_datagen = ImageDataGenerator(featurewise_center=False,
    #                                    samplewise_center=False,
    #                                    featurewise_std_normalization=False,
    #                                    samplewise_std_normalization=False,
    #                                    zca_whitening=False,
    #                                    rotation_range=0,
    #                                    width_shift_range=0.125,
    #                                    height_shift_range=0.125,
    #                                    horizontal_flip=True,
    #                                    vertical_flip=False,
    #                                    fill_mode='nearest')
    # train_gen, val_gen = _get_data_generators(train_datagen)
    callbacks = _get_callbacks(config.get_top_model_weights_path())
    test_datagen = ImageDataGenerator()
    model.fit_generator(
        datagen.flow(X_train, Y_train, shuffle=True),
        samples_per_epoch=X_train.shape[0],
        nb_epoch=train_top_layers_nb_epoch,
        validation_data=test_datagen.flow(X_test, Y_test),
        nb_val_samples=X_test.shape[0],
        callbacks=callbacks)
    return model


def fine_tune_top_2_inception_blocks(model, X_train, Y_train, X_test, Y_test, datagen):
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    # print("Compiling model...")
    model.compile(optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=["accuracy"])

    # train_datagen = ImageDataGenerator(
    #     featurewise_center=False,
    #     samplewise_center=False,
    #     featurewise_std_normalization=False,
    #     samplewise_std_normalization=False,
    #     zca_whitening=False,
    #     # rotation_range=0,
    #     width_shift_range=0.125,
    #     height_shift_range=0.125,
    #     horizontal_flip=True,
    #     vertical_flip=False,
    #     fill_mode='nearest',
    #     rotation_range=30., shear_range=0.2, zoom_range=0.2,
    #     # horizontal_flip=True
    # )
    # train_gen, val_gen = _get_data_generators(train_datagen)
    callbacks = _get_callbacks(config.get_fine_tuned_weights_path(), patience=30)
    test_datagen = ImageDataGenerator()

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        datagen.flow(X_train, Y_train, shuffle=True),
        samples_per_epoch=X_train.shape[0],
        nb_epoch=fine_tune_nb_epoch,
        validation_data=test_datagen.flow(X_test, Y_test),
        nb_val_samples=X_test.shape[0],
        callbacks=callbacks)

    model.save(config.get_model_path())


def train():
    X, y, tags = dataset(config.train_dir, 224)
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count * 4 // 5
    X_train = X[:train_size]
    y_train = y[:train_size]
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    X_test = X[train_size:]
    y_test = y[train_size:]
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    datagen.fit(X_train)

    model = create()
    model = train_top_layers(model, X_train, Y_train, X_test, Y_test, datagen)
    fine_tune_top_2_inception_blocks(model, X_train, Y_train, X_test, Y_test, datagen)

    util.save_classes(tags)


def load_trained():
    model = load_model(config.get_model_path())
    util.load_classes()
    return model


def load_img(img_path):
    n = 224
    img = scipy.misc.imread(img_path, mode='RGB')
    height, width, chan = img.shape
    centery = height // 2
    centerx = width // 2
    radius = min((centerx, centery))
    img = img[centery - radius:centery + radius, centerx - radius:centerx + radius]
    img = scipy.misc.imresize(img, size=(n, n), interp='bilinear')
    x = preprocess_input(img)
    x = x.transpose((2, 0, 1))
    return x
