#!/usr/bin/env python
import os
import glob
import tarfile
import numpy as np
from scipy.io import loadmat
from shutil import copyfile, rmtree
import sys
import config

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

data_path = 'data'


def download_file(url, dest=None):
    if not dest:
        dest = os.path.join(data_path, url.split('/')[-1])
    urlretrieve(url, dest)


# Download the Oxford102 dataset into the current directory
if not os.path.exists(data_path):
    os.mkdir(data_path)

flowers_archive_path = os.path.join(data_path, '102flowers.tgz')
if not os.path.isfile(flowers_archive_path):
    print ('Downloading images...')
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')
tarfile.open(flowers_archive_path).extractall(path=data_path)

image_labels_path = os.path.join(data_path, 'imagelabels.mat')
if not os.path.isfile(image_labels_path):
    print("Downloading image labels...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')

setid_path = os.path.join(data_path, 'setid.mat')
if not os.path.isfile(setid_path):
    print("Downloading train/test/valid splits...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')

# Read .mat file containing training, testing, and validation sets.
setid = loadmat(setid_path)

idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

# Read .mat file containing image labels.
image_labels = loadmat(image_labels_path)['labels'][0]

# Subtract one to get 0-based labels
image_labels -= 1

files = sorted(glob.glob(os.path.join(data_path, 'jpg', '*.jpg')))
labels = np.array([i for i in zip(files, image_labels)])

# Get current working directory for making absolute paths to images
cwd = os.path.dirname(os.path.realpath(__file__))

if os.path.exists(config.data_dir):
    rmtree(config.data_dir, ignore_errors=True)
os.mkdir(config.data_dir)


def move_files(dir_name, labels):
    cur_dir_path = os.path.join(config.data_dir, dir_name)
    if not os.path.exists(cur_dir_path):
        os.mkdir(cur_dir_path)

    for i in range(0, 102):
        class_dir = os.path.join(config.data_dir, dir_name, str(i))
        os.mkdir(class_dir)

    for label in labels:
        src = str(label[0])
        dst = os.path.join(cwd, config.data_dir, dir_name, label[1], src.split(os.sep)[-1])
        copyfile(src, dst)


move_files('train', labels[idx_test, :])
move_files('test', labels[idx_train, :])
move_files('valid', labels[idx_valid, :])
