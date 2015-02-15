# caffe-oxford102
This is for training a deep convolutional neural network to classify images in the Oxford 102 category flower dataset.

Download the Oxford 102 category dataset:

`./get_oxford102.sh`

Create the Caffe style training and testing set files:

`./create_caffe_splits.py`

Assuming you have Caffe installed in `CAFFE_HOME`, copy over the test.txt and train.txt files to there:

```
mkdir $CAFFE_HOME/data/oxford102
cp data/train.txt data/test.txt $CAFFE_HOME/data/oxford102
```


