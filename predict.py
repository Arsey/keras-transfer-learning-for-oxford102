import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import config
import util

tic = time.clock()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--check_relativity', action='store_true')
    parser.add_argument('--data_dir', help='Path to data train directory')
    args = parser.parse_args()
    return args


args = parse_args()

print('Called with args:')
print(args)

img_path = args.path
show_accuracy = args.accuracy
show_time = args.execution_time
data_dir = args.data_dir

if data_dir:
    config.data_dir = data_dir
    config.set_paths()

config.classes = util.get_classes_from_train_dir()
classes_in_keras_format = util.get_classes_in_keras_format()


def get_inputs_and_trues(im_path):
    if os.path.isdir(im_path):
        files = glob.glob(im_path + '*.jpg')
    elif im_path.find('*') > 0:
        files = glob.glob(im_path)
    else:
        files = [im_path]

    inputs = []
    y_true = []

    for i in files:
        x = util.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    if not inputs:
        print('No images found by the given path')
        exit(1)
    return y_true, inputs, files


def predict(path):
    model = util.load_model(nb_class=len(config.classes))
    model.load_weights(config.fine_tuned_weights_path)

    y_true, inputs, files = get_inputs_and_trues(path)

    if args.store_activations:
        util.save_activations(model, inputs, files)
    if args.check_relativity:
        af = util.get_activation_function(model, 'fc2')
        acts = util.get_activations(af, [inputs[0]])
        relativity_clf = joblib.load(config.relativity_model_path)
        predicted_relativity = relativity_clf.predict(acts)[0]
        print(relativity_clf.__classes[predicted_relativity])

    out = model.predict(np.array(inputs))
    predictions = np.argmax(out, axis=1)

    for i, p in enumerate(predictions):
        recognized_class = classes_in_keras_format.keys()[classes_in_keras_format.values().index(p)]
        print '{} ({}) ---> {} ({})'.format(y_true[i], files[i].split(os.sep)[-2], p, recognized_class)

    if show_accuracy:
        print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions))


predict(img_path)

if show_time:
    toc = time.clock()
    print 'Time: %s' % (toc - tic)
