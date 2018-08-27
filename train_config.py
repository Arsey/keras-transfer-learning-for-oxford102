from os.path import join as join_path
import os

abspath = os.path.dirname(os.path.abspath(__file__))

data_dir = join_path(abspath, 'data/sorted')
trained_dir = join_path(abspath, 'trained')

train_dir, validation_dir = None, None

MODEL_VGG16 = 'vgg16'
MODEL_INCEPTION_V3 = 'inception_v3'
MODEL_RESNET50 = 'resnet50'
MODEL_RESNET152 = 'resnet152'

model = MODEL_RESNET50

plots_dir = join_path(abspath, 'plots')

# server settings
server_address = ('0.0.0.0', 8181)

classes = []

nb_train_samples = 0
nb_validation_samples = 0


def set_paths():
    global train_dir, validation_dir
    train_dir = join_path(data_dir, 'train/')
    validation_dir = join_path(data_dir, 'valid/')


set_paths()


def get_fine_tuned_weights_path(checkpoint=False):
    return join_path(trained_dir, 'weights{}.h5').format('-checkpoint' if checkpoint else '')


def get_novelty_detection_model_path():
    return join_path(trained_dir, 'novelty_detection-model')


def get_compiled_model_path():
    return join_path(trained_dir, 'model-compiled.json')


def get_model_path():
    return join_path(trained_dir, 'model.h5')


def get_classes_path():
    return join_path(trained_dir, 'classes')


def get_activations_path():
    return join_path(trained_dir, 'activations.csv')
