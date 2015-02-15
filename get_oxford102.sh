#!/usr/bin/env sh

# This script downloads the Oxford 102 category flower dataset including:
# - jpg images
# - images labels
# - the training/testing/validation splits

mkdir data
cd data
mkdir oxford102

echo "Downloading flower images..."

wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

tar -xf 102flowers.tgz -C oxford102

echo "Downloading image labels..."

wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

echo "Downloading data splits..."

wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat

cd ../

