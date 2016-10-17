import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score

import config
import util

tic = time.clock()
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', help='Path to image that should be predicted by the model')
parser.add_argument('-e', '--epoch', help='Number epochs')
parser.add_argument('-a', '--accuracy', action='store_true', help='To print accuracy score')
parser.add_argument('-t', '--time', action='store_true')

args = parser.parse_args()

img_path = args.path
nb_epoch = args.epoch
show_accuracy = args.accuracy
show_time = args.time


def get_inputs_and_trues(path):
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
    elif path.find('*'):
        files = glob.glob(path)
    else:
        files = [path]
    inputs = []
    y_true = []
    for i in files:
        x = util.load_img(i)
        y_true.append(int(i.split(os.sep)[3]))
        inputs.append(x)
    return y_true, inputs


def predict(path):
    classes = util.get_classes_from_train_dir()

    model = util.load_model(nb_class=len(classes))
    model.load_weights(config.fine_tuned_weights_path)

    y_true, inputs = get_inputs_and_trues(path)

    out = model.predict(np.array(inputs))
    predictions = np.argmax(out, axis=1)

    class_indices = dict(zip(classes, range(len(classes))))
    keys = class_indices.keys()
    values = class_indices.values()
    for i, p in enumerate(predictions):
        predictions[i] = keys[values.index(p)]

    print predictions[0] if len(predictions) == 1 else predictions

    if show_accuracy:
        print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions))


predict(img_path)

if show_time:
    toc = time.clock()
    print 'Time: %s' % (toc - tic)
