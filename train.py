import util
import config
import numpy as np
import argparse
import os
import traceback

np.random.seed(1337)  # for reproducibility


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to data dir')
    parser.add_argument('--model', type=str, default=config.MODEL_VGG16, help='Base model architecture')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()

        if args.data_dir:
            config.data_dir = args.data_dir
            config.set_paths()
        if args.model:
            config.model = args.model

        util.lock()

        util.override_keras_directory_iterator_next()
        config.classes = util.get_classes_from_train_dir()

        # set samples info
        samples_info = util.get_samples_info()
        config.nb_train_samples = samples_info[config.train_dir]
        config.nb_validation_samples = samples_info[config.validation_dir]

        if not os.path.exists(config.trained_dir):
            os.mkdir(config.trained_dir)

        # train
        model_module = util.get_model_module()
        model_module.train()
        print('Training is finished!')
    except (KeyboardInterrupt, SystemExit):
        util.unlock()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    util.unlock()
