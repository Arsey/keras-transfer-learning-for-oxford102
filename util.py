import matplotlib

matplotlib.use('Agg')  # fixes issue if no GUI provided

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import importlib

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
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


def get_model_module():
    return importlib.import_module("models.{}".format(config.model))


def get_activation_function(m, layer):
    x = [m.layers[0].input, K.learning_phase()]
    y = [m.get_layer(layer).output]
    return K.function(x, y)


def get_activations(activation_function, X_batch):
    activations = activation_function([X_batch, 0])
    return activations[0][0]


def save_activations(model, inputs, files,layer):
    all_activations = []
    ids = []
    af = get_activation_function(model, layer)
    for i in range(len(inputs)):
        acts = get_activations(af, [inputs[i]])
        all_activations.append(acts)
        ids.append(files[i].split('/')[-2])

    submission = pd.DataFrame(all_activations)
    submission.insert(0, 'class', ids)
    submission.reset_index()
    submission.to_csv(config.activations_path, index=False)
