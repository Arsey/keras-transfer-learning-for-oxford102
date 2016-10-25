data_dir = 'data/sorted'

train_dir = '{}/train/'.format(data_dir)
validation_dir = '{}/valid/'.format(data_dir)

bf_train_path = 'trained/bottleneck_features_train.npy'
bf_valid_path = 'trained/bottleneck_features_validation.npy'
top_model_weights_path = 'trained/top-model-weights.h5'
fine_tuned_weights_path = 'trained/fine-tuned-vgg16-weights.h5'

plots_dir = 'plots'

img_size = (224, 224)

# server settings
server_address = ('127.0.0.1', 4444)
buffer_size = 4096

classes = []

nb_train_samples = 0
nb_validation_samples = 0
