data_dir = 'data/sorted'

train_data_dir = '{}/train/'.format(data_dir)
validation_data_dir = '{}//valid/'.format(data_dir)

bf_train_path = 'trained/bottleneck_features_train.npy'
bf_valid_path = 'trained/bottleneck_features_validation.npy'
top_model_weights_path = 'trained/bottleneck_fc_model(nb_epoch={},output_dim={}).h5'
fine_tuned_weights_path = 'trained/fine-tuned-vgg16-weights.h5'

plots_dir = 'plots'

img_size = (224, 224)

nb_classes = 102

info_file_path = '{}/info.csv'.format(data_dir)

output_dim = 4096
