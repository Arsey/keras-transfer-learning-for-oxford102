#!/usr/bin/env python
import os
import glob
import urllib
import tarfile
import numpy as np
from scipy.io import loadmat
from shutil import copyfile, rmtree

import config


def download_file(url, dest=None):
    if not dest:
        dest = 'data/' + url.split('/')[-1]
    urllib.urlretrieve(url, dest)


# Download the Oxford102 dataset into the current directory
if not os.path.exists('data'):
    os.mkdir('data')

if not os.path.isfile('data/102flowers.tgz'):
    print ('Downloading images...')
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')
tarfile.open('data/102flowers.tgz').extractall(path='data/')

if not os.path.isfile('data/imagelabels.mat'):
    print("Downloading image labels...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')

if not os.path.isfile('data/setid.mat'):
    print("Downloading train/test/valid splits...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')

# Read .mat file containing training, testing, and validation sets.
setid = loadmat('data/setid.mat')

idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

# Read .mat file containing image labels.
image_labels = loadmat('data/imagelabels.mat')['labels'][0]

# Subtract one to get 0-based labels
image_labels -= 1

files = sorted(glob.glob('data/jpg/*.jpg'))
labels = np.array(zip(files, image_labels))

# Get current working directory for making absolute paths to images
cwd = os.path.dirname(os.path.realpath(__file__))

if os.path.exists(config.data_dir):
    rmtree(config.data_dir, ignore_errors=True)
os.mkdir(config.data_dir)


def move_files(dir_name, labels):
    if not os.path.exists('{}/{}'.format(config.data_dir, dir_name)):
        os.mkdir('{}/{}'.format(config.data_dir, dir_name))

    for i in range(0, 102):
        os.mkdir('{}/{}/{}'.format(config.data_dir, dir_name, i))

    for label in labels:
        pic_path = str(label[0])
        copyfile(pic_path,
                 '{}/{}/{}/{}/{}'.format(cwd, config.data_dir, dir_name, label[1], pic_path.split(os.sep)[-1]))


move_files('train', labels[idx_test, :])
move_files('test', labels[idx_train, :])
move_files('valid', labels[idx_valid, :])
