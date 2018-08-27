FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

MAINTAINER Alexander Lazarev <arseysensector@gmail.com>

ENV LIB_NAME="easyclassify"

ARG KERAS_VERSION=2.1.3
ARG TENSORFLOW_VERSION=0.12.1
ARG TENSORFLOW_ARCH=gpu

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    python-dev \
    python-tk \
    python-numpy \
    python3-dev \
    python3-tk \
    python3-numpy

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Add SNI support to Python
RUN pip --no-cache-dir install \
		pyopenssl \
		ndg-httpsclient \
		pyasn1

# Install useful Python packages using apt-get to avoid version incompatibilities with Tensorflow binary
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/tensorflow/issues/2034)
RUN apt-get update && apt-get install -y \
		python-numpy \
		python-scipy \
		python-nose \
		python-h5py \
		python-skimage \
		python-matplotlib \
		python-pandas \
		python-sklearn \
		python-sympy 	\
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*


# Install other useful Python packages using pip
RUN pip --no-cache-dir install \
		Cython \
		Pillow \
		six \
		flask


# Install TensorFlow
RUN pip --no-cache-dir install \
	https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow_${TENSORFLOW_ARCH}-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl --ignore-installed numpy

# Install Keras
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION} --ignore-installed six


WORKDIR /opt

RUN git clone https://github.com/Arsey/keras-transfer-learning-for-oxford102.git /opt/$LIB_NAME

ADD https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 /root/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

RUN pip --no-cache-dir install \
    Flask-WTF

RUN pip install gevent
RUN pip install psutil
RUN pip install Flask-WTF
RUN pip install Flask-SocketIO==2.6
