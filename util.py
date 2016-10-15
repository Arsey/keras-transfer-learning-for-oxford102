import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from keras.preprocessing import image
import config
import os


def save_history(history, prefix, lr=None, output_dim=None, nb_epoch=None, img_size=None):
    if 'acc' not in history.history:
        return

    img_path = '{}/{}-%s(lr={}, output_dim={}, nb_epoch={}, img_size={}).jpg'.format(
        config.plots_dir,
        prefix,
        lr,
        output_dim,
        nb_epoch,
        img_size)

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


def get_samples_info():
    info = np.genfromtxt(config.info_file_path, delimiter=',')
    nb_train_samples = int(info[0])
    nb_validation_samples = int(info[2])

    return nb_train_samples, nb_validation_samples


def load_model(weights_path=None):
    from keras.models import Model
    from keras.layers import Flatten, Dense, Input
    from keras.applications.vgg16 import VGG16

    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,) + config.img_size))

    x = base_model.output
    x = Flatten()(x)

    if weights_path:
        weights_file = h5.File(os.path.dirname(os.path.abspath(__file__)) + '/' + weights_path)

        g = weights_file['dense_1']
        weights = [g[p] for p in g]
        x = Dense(config.output_dim, activation='relu', weights=weights)(x)

        g = weights_file['dense_2']
        weights = [g[p] for p in g]
        x = Dense(config.output_dim, activation='relu', weights=weights)(x)

        g = weights_file['dense_3']
        weights = [g[p] for p in g]
        predictions = Dense(config.nb_classes, activation='softmax', weights=weights)(x)

        weights_file.close()

    else:
        x = Dense(config.output_dim, activation='relu')(x)
        x = Dense(config.output_dim, activation='relu')(x)
        predictions = Dense(config.nb_classes, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)

    return model


def load_img(path):
    img = image.load_img(path, target_size=config.img_size)
    x = image.img_to_array(img)
    x *= 1. / 255
    return x


def get_numbered_classes(num):
    return [str(x) for x in range(num)]
