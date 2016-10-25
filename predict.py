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
parser.add_argument('-a', '--accuracy', action='store_true', help='To print accuracy score')
parser.add_argument('-e', '--execution_time', action='store_true')
parser.add_argument('-t', '--train_dir', help='Path to data train directory')

args = parser.parse_args()

img_path = args.path
show_accuracy = args.accuracy
show_time = args.execution_time
train_dir = args.train_dir

if train_dir:
    config.train_dir = train_dir

config.classes = util.get_classes_from_train_dir()
classes_in_keras_format = util.get_classes_in_keras_format()


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
        image_class = i.split(os.sep)[-2]
        keras_class = int(classes_in_keras_format[image_class])
        y_true.append(keras_class)
        inputs.append(x)
    return y_true, inputs, files


def predict(path):
    model = util.load_model(nb_class=len(config.classes))
    model.load_weights(config.fine_tuned_weights_path)

    y_true, inputs, files = get_inputs_and_trues(path)

    out = model.predict(np.array(inputs))
    predictions = np.argmax(out, axis=1)

    for i, p in enumerate(predictions):
        recognized_class = classes_in_keras_format.keys()[classes_in_keras_format.values().index(p)]

        print '{} ({}) --->>> {} ({})'.format(y_true[i], files[i].split(os.sep)[-2], p, recognized_class)

    if show_accuracy:
        print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions))


predict(img_path)

if show_time:
    toc = time.clock()
    print 'Time: %s' % (toc - tic)
