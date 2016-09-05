import argparse
import os
import numpy as np
import time

start_time = time.time()

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
import config
import class_labels

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to image that should be predicted by the model')
args = parser.parse_args()
img_path = args.path

assert os.path.exists(img_path), True

model = VGG16(weights=None, include_top=True)
model.load_weights(config.fine_tuned_weights_path)
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True))


def predict(path):
    files = [path]
    if os.path.isdir(path):
        files = os.listdir(path)

    for i in files:
        st = time.time()
        img = load_img(path + i, target_size=config.img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        out = model.predict(x)
        predicted_class = np.argmax(out[0])
        print '{} - {}'.format(predicted_class, class_labels.labels[predicted_class])
        print('--- %s to predict %s' % (time.time() - st, i))
        print('*' * 100)


print('---- %s before predict ----' % (time.time() - start_time))
start_time = time.time()
predict(img_path)
