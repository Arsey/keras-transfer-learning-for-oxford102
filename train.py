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
        util.set_classes_from_train_dir()
        util.set_samples_info()
        class_weight = util.get_class_weight(config.train_dir)

        if not os.path.exists(config.trained_dir):
            os.mkdir(config.trained_dir)

        # train
        model_module = util.get_model_module()
        model_module.train(class_weight=class_weight)
        print('Training is finished!')
    except (KeyboardInterrupt, SystemExit):
        util.unlock()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    util.unlock()
