FROM python:3.6
MAINTAINER William Silversmith
# This image contains private keys, make sure the image is not pushed to docker hub or any public repo.
## INSTALL gsutil
# Prepare the image.
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
    curl \
    git \
    openssh-client \
    python-openssl \
    python \
    python-pip \
    python-dev \
#    python-h5py \
    python-numpy \
    python-setuptools \
    libboost-all-dev \
#    libhdf5-dev \
    liblzma-dev \
    libgmp-dev \
#    libmpfr-dev \
#    libxml2-dev \
    screen \
    software-properties-common \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install setuptools Cython wheel numpy 

# install neuroglancer
RUN mkdir /.ssh
ADD ./ /igneous
RUN pip install -r /igneous/requirements.txt \
    && pip install pyasn1 --upgrade \
    && cd /igneous && pip install -e .

CMD python /igneous/igneous/task_execution.py





