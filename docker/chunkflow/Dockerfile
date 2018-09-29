FROM seunglab/chunkflow:latest 
MAINTAINER Jingpeng Wu
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
    git \
    wget \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install setuptools Cython wheel numpy 

# install igneous 
RUN mkdir /.ssh
WORKDIR /
RUN git clone --depth 1 --single-branch -b chunkflow https://github.com/seung-lab/igneous.git 
#ADD ./ /igneous
RUN pip install -r /igneous/requirements.txt 
RUN pip install pyasn1 --upgrade
RUN cd /igneous && pip install -e .

#CMD python /igneous/igneous/task_execution.py





