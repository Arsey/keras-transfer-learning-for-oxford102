# Keras pretrained models (currently only VGG16) + Transfer Learning for predicting classes in the Oxford 102 flower dataset

This bootstraps the training of deep convolutional neural networks with [Keras](https://keras.io/) to classify images in the [Oxford 102 category flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

**Train process is fully automated and best weights for model are saving**


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
python train.py
```

### Step 3: Get predictions using `predict.py` or `server.py` + `client.py` 

Using `predict.py`:
```
python predict.py -p "data/sorted/test/24/image_06592.jpg"
```

Using `server.py` + `client.py`:

1. run server and wait till model is loaded. Do not break server, it should be run and listen for incoming connections
```
python server.py
```
2. send requests using client
```
python client.py -p "path/to/image"
```