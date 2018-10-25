FROM seunglab/chunkflow:latest 
MAINTAINER Jingpeng Wu
# This image contains private keys, make sure the image is not pushed to docker hub or any public repo.
## INSTALL gsutil
# Prepare the image.
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
    software-properties-common \
    curl \
    git \
    openssh-client \
    python3-openssl \
    python3 \
    python3-pip \
    python3-dev \
#    python-h5py \
    python3-numpy \
    python3-setuptools \
    libboost-all-dev \
#    libhdf5-dev \
    liblzma-dev \
    libgmp-dev \
#    libmpfr-dev \
#    libxml2-dev \
    screen \
    unzip \
    vim \
    git \
    wget \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pip -U
RUN pip3 install setuptools -U
RUN pip3 install pyasn1 --upgrade
RUN pip3 install Cython wheel numpy 

# install igneous 
RUN mkdir /.ssh
WORKDIR /
RUN git clone --depth 1 --single-branch -b chunkflow https://github.com/seung-lab/igneous.git 
#ADD ./ /igneous
RUN pip3 install -r /igneous/requirements.txt 

WORKDIR /igneous 
RUN pip3 install intern -U
RUN pip3 install -e .

#CMD python /igneous/igneous/task_execution.py





