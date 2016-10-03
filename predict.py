import time

tic = time.clock()

import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score

import config
import util

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', help='Path to image that should be predicted by the model')
parser.add_argument('-e', '--epoch', help='Number epochs')
parser.add_argument('-b', '--bottleneck', action='store_true',
                    help='Uses bottlenecks features to predict instead of final fine tuned VGG16 weights')
parser.add_argument('-a', '--accuracy', action='store_true', help='To print accuracy score')
parser.add_argument('-t', '--time', action='store_true')
parser.add_argument('-r', '--roc', action='store_true',
                    help='Create plot with ROC(Receiver Operating Characteristic) curves')

args = parser.parse_args()

img_path = args.path
nb_epoch = args.epoch
use_bottleneck = args.bottleneck
show_accuracy = args.accuracy
show_time = args.time
show_time = args.roc


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
    model = util.load_model()
    model.load_weights(config.fine_tuned_weights_path)

    y_true, inputs = get_inputs_and_trues(path)

    out = model.predict(np.array(inputs))
    predictions = np.argmax(out, axis=1)

    print predictions[0] if len(predictions) == 1 else predictions

    if show_accuracy:
        print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions))


def predict_from_bottleneck(path):
    weight_path = config.top_model_weights_path.format(nb_epoch, config.output_dim)
    model = util.load_model(weight_path)

    y_true, inputs = get_inputs_and_trues(path)

    out = model.predict(np.array(inputs))
    predictions = np.argmax(out, axis=1)

    print predictions[0] if len(predictions) == 1 else predictions

    if show_accuracy:
        print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions))


if not use_bottleneck:
    predict(img_path)
else:
    predict_from_bottleneck(img_path)

if show_time:
    toc = time.clock()
    print 'Time: %s' % (toc - tic)
