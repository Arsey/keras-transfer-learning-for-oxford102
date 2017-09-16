from __future__ import print_function

import os
import argparse
import traceback
import numpy as np

import util
import config
import keras
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(1337)  # for reproducibility
batch_size=16

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to data dir')
    parser.add_argument('--trained_file', default='./trained/model-resnet50.h5', help='Path to trained file')
    parser.add_argument('--model', type=str, default='resnet50', help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16])
    args = parser.parse_args()
    config.model = args.model
    config.trained_file = args.trained_file
    return args


def init():
    util.set_img_format()
    util.set_classes_from_train_dir()
    util.set_samples_info()
    if not os.path.exists(config.trained_file):
        raise Exception('trained_file not exists')

def test():
    img_size = (224, 224)

    print("Creating model...")

    model = keras.models.load_model(config.trained_file)

    print("Model is created")

    idg = ImageDataGenerator()
    idg.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
    test_generator = idg.flow_from_directory(config.test_dir,
        batch_size=batch_size,
        target_size=img_size,
        classes=config.classes)

    # -- Evaluate generator -- #
    result = model.evaluate_generator(
        generator=test_generator,
        steps=config.nb_test_samples)

    print("Model [loss, accuracy]: {0}".format(result))

    # -- Predict generator -- #
    predict = model.predict_generator(
        generator=test_generator,
        steps=config.nb_test_samples)

    print("model predictions: {0}".format(predict))
    print('Testing is finished!')


if __name__ == '__main__':
    try:
        args = parse_args()
        if args.data_dir:
            config.data_dir = args.data_dir
            config.set_paths()
        if args.model:
            config.model = args.model

        init()
        test()

    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        util.unlock()
