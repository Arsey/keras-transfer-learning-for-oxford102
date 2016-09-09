import matplotlib.pyplot as plt
import numpy as np

import config


def save_history(history, prefix, lr=None, output_dim=None, nb_epoch=None, img_size=None):
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
