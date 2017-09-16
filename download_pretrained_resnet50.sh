#!/bin/sh

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

cd $DIR
mkdir trained
cd trained

HOST='https://drop.jackhftang.com/storage'
curl -LO "$HOST/model-resnet50.h5"
curl -LO "$HOST/fine-tuned-resnet50-weights.h5"
curl -LO "$HOST/classes-resnet50"

