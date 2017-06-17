import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import keras

# will take the video memory as much as needed for the chosen model
os.environ["THEANO_FLAGS"] = "lib.cnmem=2"
keras.backend.set_image_dim_ordering('th')

import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3, config.MODEL_VGG16])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model_module.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    return y_true, inputs


def predict(path):
    files = get_files(path)
    n_files = len(files)
    print('Found {} files'.format(n_files))

    if args.novelty_detection:
        activation_function = util.get_activation_function(model, model_module.noveltyDetectionLayerName)
        novelty_detection_clf = joblib.load(config.get_novelty_detection_model_path())

    y_trues = []
    predictions = np.zeros(shape=(n_files,))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)

        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true

        if args.store_activations:
            util.save_activations(model, inputs, files[n_from:n_to], model_module.noveltyDetectionLayerName, n)

        if args.novelty_detection:
            activations = util.get_activations(activation_function, [inputs[0]])
            nd_preds = novelty_detection_clf.predict(activations)[0]
            print(novelty_detection_clf.__classes[nd_preds])

        if not args.store_activations:
            # Warm up the model
            if n == 0:
                print('Warming up the model')
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                print('Warming up took {} s'.format(end - start))

            # Make predictions
            start = time.clock()
            out = model.predict(np.array(inputs))
            end = time.clock()
            predictions[n_from:n_to] = np.argmax(out, axis=1)
            print 'Prediction on batch {} took: {}'.format(n, end - start)

    if not args.store_activations:
        for i, p in enumerate(predictions):
            recognized_class = classes_in_keras_format.keys()[classes_in_keras_format.values().index(p)]
            print '| should be {} ({}) -> predicted as {} ({})'.format(y_trues[i], files[i].split(os.sep)[-2], p,
                                                                       recognized_class)

        if args.accuracy:
            print 'Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions))

        if args.plot_confusion_matrix:
            cnf_matrix = confusion_matrix(y_trues, predictions)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=False)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=True)


if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('=' * 50)
    print('Called with args:')
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    model_module = util.get_model_class_instance()
    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    predict(args.path)

    if args.execution_time:
        toc = time.clock()
        print 'Time: %s' % (toc - tic)
