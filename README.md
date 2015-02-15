# caffe-oxford102
This is for training a deep convolutional neural network to classify images in the Oxford 102 category flower dataset.

Download the Oxford 102 category dataset:

`./get_oxford102.sh`

Set the environment variable `CAFFE_HOME` to point to your installation of Caffe, then create the Caffe style training and testing set files:

`./create_caffe_splits.py`


