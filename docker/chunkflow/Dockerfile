FROM seunglab/chunkflow:latest 
MAINTAINER Jingpeng Wu

## INSTALL gsutil
# Prepare the image.
ENV DEBIAN_FRONTEND noninteractive

# install igneous 
WORKDIR /
RUN git clone --depth 1 --single-branch -b chunkflow https://github.com/seung-lab/igneous.git 

RUN apt-get update \
    # Build dependencies
    && apt-get install -y -qq --no-install-recommends \
        git \
        libboost-dev \
        build-essential \
    # igneous + runtime dependencies
    && cd igneous \
    && pip3 install --no-cache-dir -r requirements.txt \
    # we can only use numpy==1.15 due to an issue of skimage:
    # https://github.com/scikit-image/scikit-image/issues/3551
    # the fix was not released yet. 
    # https://github.com/scikit-image/scikit-image/pull/3556
    && pip3 install numpy==1.15 --no-cache-dir \
    && pip3 install --no-cache-dir -e . \
    \
    # Cleanup build dependencies
    && apt-get remove --purge -y \
        libboost-dev \
        build-essential \
    && apt-get autoremove --purge -y \
    # Cleanup apt
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Cleanup temporary python files
    && find /opt/conda/bin/python3.6 -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +

#CMD python /igneous/igneous/task_execution.py
