import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score
import h5py as h5

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array, load_img
import config

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to image that should be predicted by the model')
args = parser.parse_args()
img_path = args.path


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
        img = load_img(i, target_size=config.img_size)
        x = img_to_array(img)
        y_true.append(int(i.split(os.sep)[3]))
        inputs.append(x)

    return y_true, inputs


def predict(path):
    model = VGG16(weights=None, include_top=True)
    model.load_weights(config.fine_tuned_weights_path)

    y_true, inputs = get_inputs_and_trues(path)

    out = model.predict(np.array(inputs))
    y_pred = np.argmax(out, axis=1)
    print y_true
    print y_pred
    print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=y_pred))


def predict_from_bottleneck(path):
    from keras.models import Model
    from keras.layers import Flatten, Dense, Input

    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(3,) + config.img_size))
    print('Model loaded.')

    x = base_model.output
    x = Flatten()(x)

    weights_file = h5.File(config.top_model_weights_path.format(20, config.output_dim))

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

    model = Model(input=base_model.input, output=predictions)

    y_true, inputs = get_inputs_and_trues(path)
    out = model.predict(np.array(inputs))
    y_pred = np.argmax(out, axis=1)
    print 'accuracy {}'.format(accuracy_score(y_true=y_true, y_pred=y_pred))


# predict(img_path)
predict_from_bottleneck(img_path)
