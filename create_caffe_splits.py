#!/usr/bin/env python
import os
import glob
import numpy as np
from scipy.io import loadmat

# Read .mat file containing training, testing, and validation sets.
setid = loadmat('data/setid.mat')

# The .mat file is 1-indexed, so we subtract one to match Caffe's convention.
idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

# Read .mat file containing image labels.
image_labels = loadmat('data/imagelabels.mat')['labels'][0]
# Subtract one to get 0-based labels
image_labels -= 1

files = sorted(glob.glob('data/oxford102/jpg/*.jpg'))
labels = np.array(zip(files, image_labels))
cwd = os.path.dirname(os.path.realpath(__file__))

def write_set_file(fout, labels):
    with open(fout, 'w+') as f:
        for label in labels:
            f.write('%s/%s %s\n' % (cwd, label[0], label[1]))

np.random.seed(777)
idx_test_perm = idx_test[np.random.permutation(len(idx_test))]
idx_train_perm = idx_train[np.random.permutation(len(idx_train))]

CAFFE_HOME = os.getenv('CAFFE_HOME', '')
if CAFFE_HOME == '':
    raise Exception('CAFFE_HOME must be set to the location of your Caffe installation.')

os.mkdir(CAFFE_HOME + '/data/oxford102')
write_set_file(CAFFE_HOME + '/data/oxford102/train.txt', labels[idx_train_perm,:])
write_set_file(CAFFE_HOME + '/data/oxford102/test.txt', labels[idx_test_perm,:])
