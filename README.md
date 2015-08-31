# caffe-oxford102
This is for training a deep convolutional neural network to classify images in the [Oxford 102 category flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). The accuracy achieved is the best I've seen.

Download the Oxford 102 category images, labels, and splits:

```bash
./get_oxford102.sh
```

Set the environment variable `CAFFE_HOME` to point to your installation of [Caffe](http://caffe.berkeleyvision.org/), then create the Caffe style training and testing set files:

```bash
./create_caffe_splits.py
```

The split file (setid.mat) lists 6,149 images in the test set and 1,020 images in the training set. We have instead trained this model on the larger set of 6,149 images and tested against the smaller set of 1,020 images.

The CNN is a fine-tuned BVLC reference CaffeNet (modified [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) trained on ILSVRC 2012). The number of outputs in the inner product layer has been set to 102 to reflect the number of flower categories. Hyperparameter choices reflect those in [Fine-tuning CaffeNet for Style Recognition on “Flickr Style” Data](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html). The global learning rate is reduced while the learning rate for the final fully connected is increased relative to the other layers.

To fit the model with initial CaffeNet's weights:

```bash
./build/tools/caffe train -solver models/oxford102/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```

After 50,000 iterations, the top-1 error is 7% on the test set of 1,020 images:
```
I0215 15:28:06.417726  6585 solver.cpp:246] Iteration 50000, loss = 0.000120038
I0215 15:28:06.417789  6585 solver.cpp:264] Iteration 50000, Testing net (#0)
I0215 15:28:30.834987  6585 solver.cpp:315]     Test net output #0: accuracy = 0.9326
I0215 15:28:30.835072  6585 solver.cpp:251] Optimization Done.
I0215 15:28:30.835083  6585 caffe.cpp:121] Optimization Done.
```

The Caffe model can be downloaded at https://s3.amazonaws.com/jgoode/oxford102.caffemodel. You can also use the Caffe utility to download this model from its [gist](https://gist.github.com/jgoode21/0179e52305ca768a601f):

```bash
./scripts/download_model_from_gist.sh 0179e52305ca768a601f <dirname>
```
Note that this uses the mean file for ILSVRC 2012 instead of the mean for the actual Oxford dataset.

The class labels were deduced by Github user [m-co](https://github.com/m-co) and can be found in the file `class-labels.py`.
