FROM python:3.6-slim
LABEL maintainer="William Silversmith, Nico Kemnitz"

ADD ./ /igneous
RUN touch projects && rm -r projects
RUN apt-get update \
    # Build dependencies
    && apt-get install -y -qq --no-install-recommends \
        git \
        libboost-dev \
        build-essential \
    # igneous + runtime dependencies
    && cd igneous \
    && pip install --no-cache-dir numpy \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . \
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
    && find /usr/local/lib/python3.6 -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +

CMD python /igneous/igneous/task_execution.py
