FROM python:3.6
MAINTAINER William Silversmith
# This image contains private keys, make sure the image is not pushed to docker hub or any public repo.

# Prepare the image.
ENV DEBIAN_FRONTEND noninteractive
ADD ./ /igneous
RUN apt update \
    # Build dependencies
    && apt install -y -qq --no-install-recommends \
        libboost-dev \
    && pip install --no-cache-dir \
        Cython \
    \
    # igneous + runtime dependencies
    && cd igneous \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . \
    \
    # Cleanup build dependencies
    && apt remove --purge -y \
        libboost-dev \
    && apt autoremove --purge -y \
    # Cleanup apt
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    # Cleanup temporary python files
    && find /usr/local/lib/python3.6 -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +

CMD python /igneous/igneous/task_execution.py
