from os.path import join as join_path
import os

abspath = os.path.dirname(os.path.abspath(__file__))

data_dir = join_path(abspath, 'data/sorted')
trained_dir = join_path(abspath, 'trained')

train_dir, validation_dir = None, None

MODEL_VGG16 = 'vgg16'
MODEL_INCEPTION_V3 = 'inception_v3'

model = MODEL_VGG16

bf_train_path = join_path(trained_dir, 'bottleneck_features_train.npy')
bf_valid_path = join_path(trained_dir, 'bottleneck_features_validation.npy')
top_model_weights_path = join_path(trained_dir, 'top-model-{}-weights.h5')
fine_tuned_weights_path = join_path(trained_dir, 'fine-tuned-{}-weights.h5')

activations_path = join_path(trained_dir, 'activations.csv')
relativity_model_path = join_path(trained_dir, 'relativity-model-{}')

plots_dir = 'plots'

img_size = (224, 224)

# server settings
server_address = ('0.0.0.0', 4444)
buffer_size = 4096

classes = []

nb_train_samples = 0
nb_validation_samples = 0


def set_paths():
    global train_dir, validation_dir
    train_dir = join_path(data_dir, 'train/')
    validation_dir = join_path(data_dir, 'valid/')


set_paths()


def get_top_model_weights_path():
    return top_model_weights_path.format(model)


def get_fine_tuned_weights_path():
    return fine_tuned_weights_path.format(model)


def get_relativity_model_path():
    return relativity_model_path.format(model)
