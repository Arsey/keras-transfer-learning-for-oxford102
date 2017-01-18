import matplotlib

matplotlib.use('Agg')  # fixes issue if no GUI provided

import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
import glob
import pandas as pd

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Dropout, Input
from keras.applications.vgg16 import VGG16
from keras import backend as K

import config


def save_history(history, prefix):
    if 'acc' not in history.history:
        return

    img_path = '{}/{}-%s.jpg'.format(config.plots_dir, prefix)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(img_path % 'accuracy')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(img_path % 'loss')
    plt.close()


def get_dir_imgs_number(dir_path):
    allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    number = 0
    for e in allowed_extensions:
        # print os.path.join(dir_path, e)
        # print len(glob.glob(os.path.join(dir_path, e)))
        number += len(glob.glob(os.path.join(dir_path, e)))
    # print 'total',number
    # exit()
    return number


def get_samples_info():
    """Walks through the train and valid directories
    and returns number of images"""
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    dirs_info = {config.train_dir: 0, config.validation_dir: 0}
    for d in dirs_info:
        iglob_iter = glob.iglob(d + '**/*.*')
        for i in iglob_iter:
            filename, file_extension = os.path.splitext(i)
            if file_extension[1:] in white_list_formats:
                dirs_info[d] += 1

    return dirs_info


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


def load_img(path):
    img = image.load_img(path, target_size=config.img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)[0]


def get_classes_from_train_dir():
    """Returns classes based on directories in train directory"""
    d = config.train_dir
    return sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])


def override_keras_directory_iterator_next():
    """Overrides .next method of DirectoryIterator in Keras
      to reorder color channels for images from RGB to BGR"""
    from keras.preprocessing.image import DirectoryIterator

    original_next = DirectoryIterator.next

    # do not allow to override one more time
    if 'custom_next' in str(original_next):
        return

    def custom_next(self):
        batch_x, batch_y = original_next(self)

        batch_x = batch_x[:, ::-1, :, :]
        return batch_x, batch_y

    DirectoryIterator.next = custom_next


def apply_mean(image_data_generator):
    """Subtracts the VGG dataset mean"""
    image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))


def get_classes_in_keras_format():
    if config.classes:
        return dict(zip(config.classes, range(len(config.classes))))
    return None


def get_activation_function(m, layer):
    x = [m.layers[0].input, K.learning_phase()]
    y = [m.get_layer(layer).output]
    return K.function(x, y)


def get_activations(activation_function, X_batch):
    activations = activation_function([X_batch, 0])
    return activations[0][0]


def save_activations(model, inputs, files):
    all_activations = []
    ids = []
    af = get_activation_function(model, 'fc2')
    for i in range(len(inputs)):
        acts = get_activations(af, [inputs[i]])
        all_activations.append(acts)
        ids.append(files[i].split('/')[-2])

    submission = pd.DataFrame(all_activations)
    submission.insert(0, 'class', ids)
    submission.reset_index()
    submission.to_csv(config.activations_path, index=False)
