# Keras pretrained models (currently VGG16 and InceptionV3) + Transfer Learning for predicting classes in the Oxford 102 flower dataset or any custom dataset

This bootstraps the training of deep convolutional neural networks with [Keras](https://keras.io/) to classify images in the [Oxford 102 category flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

**Train process is fully automated and the best weights for the model will be saved.**

**This code can be used for any dataset, just follow the original files structure in data/sorted directory after running `bootstrap.py`. If you wish to store your dataset somewhere else, you can do it and run `train.py` with setting a path to dataset with a special parameter `--data_dir==path/to/your/sorted/data`**


## Overview

* `bootstrap.py`: to download the Oxford 102 dataset and prepare image files
* `train.py`: starts training process end-to-end
* `server.py`: a small python server based on sockets and designed to keep a model in memory for fast recognition requests
* `client.py`: a client that sends requests to server.py


## Usage

### Step 1: Bootstrap
```
python bootstrap.py
```

### Step 2: Train
```
python train.py --model=vgg16
```
or
```
python train.py --model=inception_v3
```

### Step 3: Get predictions using `predict.py` or `server.py` + `client.py` 

Using `predict.py`:
```
python predict.py -p "/path/to/image" --model=vgg16
```
or
```
python predict.py -p "/path/to/image" --model=inception_v3
```

Using `server.py` + `client.py`:

1. run server and wait till model is loaded. Do not break server, it should be run and listen for incoming connections
```
python server.py
```
2. send requests using client
```
python client.py -p "/path/to/image"
```