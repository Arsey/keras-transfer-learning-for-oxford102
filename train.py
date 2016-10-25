import bottleneck
import fine_tuning
import util
import config
import numpy as np
import argparse

np.random.seed(1337)  # for reproducibility


def register_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train_dir', help='Path to data train directory')
    parser.add_argument('-v', '--valid_dir', help='Path to data validation directory')

    return parser.parse_args()


args = register_args()
train_dir = args.train_dir
valid_dir = args.valid_dir

if valid_dir or train_dir:
    config.train_dir = train_dir
    config.validation_dir = valid_dir

util.override_keras_directory_iterator_next()
config.classes = util.get_classes_from_train_dir()

# set samples info
samples_info = util.get_samples_info()
config.nb_train_samples = samples_info[config.train_dir]
config.nb_validation_samples = samples_info[config.validation_dir]

# train
bottleneck.save_bottleneck_features()
bottleneck.train_top_model()
fine_tuning.tune()
