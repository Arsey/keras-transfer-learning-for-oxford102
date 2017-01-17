import bottlenecks
import fine_tuning
import util
import config
import numpy as np
import argparse

np.random.seed(1337)  # for reproducibility


def register_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', help='Path to data dir')

    return parser.parse_args()


args = register_args()
data_dir = args.data_dir

if data_dir:
    config.data_dir = data_dir

config.set_paths()

util.override_keras_directory_iterator_next()
config.classes = util.get_classes_from_train_dir()

# set samples info
samples_info = util.get_samples_info()
config.nb_train_samples = samples_info[config.train_dir]
config.nb_validation_samples = samples_info[config.validation_dir]

# train
bottlenecks.save_bottleneck_features()
bottlenecks.train_top_model()
fine_tuning.tune()

print('Train is finished!')
