## See my application for identifying plants - [Plant Care](https://plants-care.com). It works using the code from the model implemented in this repo.

# Keras pretrained models (VGG16, InceptionV3, Resnet50, Resnet152) + Transfer Learning for predicting classes in the Oxford 102 flower dataset (or any custom dataset)

This bootstraps the training of deep convolutional neural networks with [Keras](https://keras.io/) to classify images in the [Oxford 102 category flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

**Train process is fully automated and the best weights will be saved.**

**The code can be used for any dataset, just follow the original files structure in data/sorted directory after running `bootstrap.py`. If you want to store your dataset somewhere else, you can do it and run `train.py` with setting a path to dataset with a special parameter `--data_dir==/full/path/to/your/sorted/data`**

**Dataset directory's structure**

![Dataset directory's structure](/imgs/data_structure.png)

*Notice:* for ResNet152 you should download weights manually [here](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6) and put them under the `project_directory/weights`


## Overview

* `bootstrap.py`: downloads the Oxford 102 dataset and prepare image files
* `train.py`: starts end-to-end training process 
* `server.py`: a small python server based on sockets and designed to keep a model in memory for fast recognition requests
* `client.py`: a client that sends requests to server.py


## Usage

### Step 1: Bootstrap
```
python bootstrap.py
```

### Step 2: Train
```
python train.py --model=resnet50
```

### Step 3: Get predictions using `predict.py` or `server.py` + `client.py` 

Using `predict.py`:
```
python predict.py --path "/full/path/to/image" --model=resnet50
```

Using `server.py` + `client.py`:

1. run server and wait till model is loaded. Do not break server, it should be run and listen for incoming connections
```
python server.py --model=resnet50
```
2. send requests using client
```
python client.py --path "/full/path/to/image"
```