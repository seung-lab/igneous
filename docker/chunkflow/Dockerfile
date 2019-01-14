FROM seunglab/chunkflow:latest 
MAINTAINER Jingpeng Wu
# This image contains private keys, make sure the image is not pushed to docker hub or any public repo.
## INSTALL gsutil
# Prepare the image.
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
#    software-properties-common \
    curl \
    git \
    openssh-client \
#    python-h5py \
    libboost-all-dev \
#    libhdf5-dev \
#    liblzma-dev \
#    libgmp-dev \
#    libmpfr-dev \
#    libxml2-dev \
#    unzip \
#    vim \
    git \
    wget \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# we can only use numpy==1.15 due to an issue of skimage:
# https://github.com/scikit-image/scikit-image/issues/3551
# the fix was not released yet. 
# https://github.com/scikit-image/scikit-image/pull/3556
RUN pip install pip setuptools pyasn1 -U --no-cache-dir \
    && pip install Cython wheel numpy==1.15 --no-cache-dir 

# install igneous 
WORKDIR /

RUN mkdir /.ssh \
    && git clone --depth 1 --single-branch -b chunkflow https://github.com/seung-lab/igneous.git \
    && pip install -r /igneous/requirements.txt --no-cache-dir 

WORKDIR /igneous 
RUN pip install intern -U --no-cache-dir \
    && pip install -e . --no-cache-dir

RUN apt-get remove -y libboost-dev
#CMD python /igneous/igneous/task_execution.py
