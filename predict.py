import time
import argparse
import os
os.environ["THEANO_FLAGS"] = "lib.cnmem=1000"
import numpy as np
import glob
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import config
import util


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
    parser.add_argument('--model', type=str, default=config.MODEL_VGG16)
    parser.add_argument('--data_dir', help='Path to data train directory')
    args = parser.parse_args()
    return args


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
    y_true, inputs, files = get_inputs_and_trues(path)

    if config.model == config.MODEL_VGG16:
        if args.store_activations:
            import train_relativity
            util.save_activations(model, inputs, files, 'fc2')
            train_relativity.train_relativity()
        if args.check_relativity:
            af = util.get_activation_function(model, 'fc2')
            acts = util.get_activations(af, [inputs[0]])
            relativity_clf = joblib.load(config.relativity_model_path)
            predicted_relativity = relativity_clf.predict(acts)[0]
            print(relativity_clf.__classes[predicted_relativity])

    if not args.store_activations:
        out = model.predict(np.array(inputs))
        predictions = np.argmax(out, axis=1)

        for i, p in enumerate(predictions):
            recognized_class = classes_in_keras_format.keys()[classes_in_keras_format.values().index(p)]
            print '{} ({}) ---> {} ({})'.format(y_true[i], files[i].split(os.sep)[-2], p, recognized_class)

        if args.accuracy:
            print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=predictions))


if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    model_module = util.get_model_module()
    model = model_module.load_trained()

    predict(args.path)

    if args.execution_time:
        toc = time.clock()
        print 'Time: %s' % (toc - tic)
