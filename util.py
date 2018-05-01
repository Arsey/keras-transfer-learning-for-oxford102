import matplotlib; matplotlib.use('Agg')  # fixes issue if no GUI provided
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import importlib
import keras
from keras import backend as K
import config
import math
import itertools


def save_history(history, prefix):
    if 'acc' not in history.history:
        return

    if not os.path.exists(config.plots_dir):
        os.mkdir(config.plots_dir)

    img_path = os.path.join(config.plots_dir, '{}-%s.jpg'.format(prefix))

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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    confusion_matrix_dir = './confusion_matrix_plots'
    if not os.path.exists(confusion_matrix_dir):
        os.mkdir(confusion_matrix_dir)

    plt.cla()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="#BFD1D4" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if normalize:
        plt.savefig(os.path.join(confusion_matrix_dir, 'normalized.jpg'))
    else:
        plt.savefig(os.path.join(confusion_matrix_dir, 'without_normalization.jpg'))


def get_dir_imgs_number(dir_path):
    allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    number = 0
    for e in allowed_extensions:
        number += len(glob.glob(os.path.join(dir_path, e)))
    return number


def set_samples_info():
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

    config.nb_train_samples = dirs_info[config.train_dir]
    config.nb_validation_samples = dirs_info[config.validation_dir]


def get_class_weight(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])
    k = 0
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))
        for i in iglob_iter:
            _, ext = os.path.splitext(i)
            if ext[1:] in white_list_formats:
                class_number[k] += 1
        k += 1

    total = np.sum(list(class_number.values()))
    max_samples = np.max(list(class_number.values()))
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(class_number[key]))
        class_weight[key] = score if score > 1. else 1.

    return class_weight


def set_classes_from_train_dir():
    """Returns classes based on directories in train directory"""
    d = config.train_dir
    config.classes = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])


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


def get_classes_in_keras_format():
    if config.classes:
        return dict(zip(config.classes, range(len(config.classes))))
    return None


def get_model_class_instance(*args, **kwargs):
    module = importlib.import_module("models.{}".format(config.model))
    return module.inst_class(*args, **kwargs)


def get_activation_function(m, layer):
    x = [m.layers[0].input, K.learning_phase()]
    y = [m.get_layer(layer).output]
    return K.function(x, y)


def get_activations(activation_function, X_batch):
    activations = activation_function([X_batch, 0])
    return activations[0][0]


def save_activations(model, inputs, files, layer, batch_number):
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
    if batch_number > 0:
        submission.to_csv(config.activations_path, index=False, mode='a', header=False)
    else:
        submission.to_csv(config.activations_path, index=False)


def lock():
    if os.path.exists(config.lock_file):
        exit('Previous process is not yet finished.')

    with open(config.lock_file, 'w') as lock_file:
        lock_file.write(str(os.getpid()))


def unlock():
    if os.path.exists(config.lock_file):
        os.remove(config.lock_file)


def is_keras2():
    return keras.__version__.startswith('2')


def get_keras_backend_name():
    try:
        return K.backend()
    except AttributeError:
        return K._BACKEND


def tf_allow_growth():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    set_session(sess)


def set_img_format():
    try:
        if K.backend() == 'theano':
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')
    except AttributeError:
        if K._BACKEND == 'theano':
            K.set_image_dim_ordering('th')
        else:
            K.set_image_dim_ordering('tf')
