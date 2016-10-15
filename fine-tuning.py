import h5py as h5
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

import util
import config

nb_epoch = 250
batch_size = 32
lr = 0.0001
nb_train_samples, nb_validation_samples = util.get_samples_info()
classes = util.get_numbered_classes(config.nb_classes)

base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,) + config.img_size))

x = base_model.output
x = Flatten(name='flatten', input_shape=base_model.output_shape[1:])(x)

weights_file = h5.File(config.top_model_weights_path.format(18, config.output_dim))

g = weights_file['dense_1']
weights = [g[p] for p in g]
x = Dense(config.output_dim, activation='relu', weights=weights, name='fc1')(x)
g = weights_file['dense_2']
weights = [g[p] for p in g]
x = Dense(config.output_dim, activation='relu', weights=weights, name='fc2')(x)
g = weights_file['dense_3']
weights = [g[p] for p in g]
predictions = Dense(config.nb_classes, activation='softmax', weights=weights, name='predictions')(x)

weights_file.close()

# set base_model layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in base_model.layers:
    layer.trainable = False

model = Model(input=base_model.input, output=predictions)

model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
    metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    config.train_data_dir,
    target_size=config.img_size,
    batch_size=batch_size,
    classes=classes)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    config.validation_data_dir,
    target_size=config.img_size,
    batch_size=batch_size,
    classes=classes)

# fine-tune the model
history = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save_weights(config.fine_tuned_weights_path)

util.save_history(history=history, prefix='fine-tuning', nb_epoch=nb_epoch, img_size=config.img_size)
