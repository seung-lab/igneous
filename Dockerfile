FROM python:3.9-slim
LABEL maintainer="William Silversmith, Nico Kemnitz"

ADD ./ /igneous

RUN apt-get update \
    # Build dependencies
    && apt-get install -y -qq --no-install-recommends \
        git \
        build-essential \
        nano \
    # igneous + runtime dependencies
    && cd igneous \
    # numpy 1.23.0 through 1.23.2 causing issues
    # with locating random module
    && pip install --no-cache-dir 'numpy<1.23.0' \
    && pip install --no-cache-dir -r requirements.txt mysql-connector-python \
    && pip install --no-cache-dir -e . \
    \
    # Cleanup build dependencies
    && apt-get remove --purge -y \
        build-essential \
    && apt-get autoremove --purge -y \
    # Cleanup apt
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Cleanup temporary python files
    && find /usr/local/lib/python3.9 -depth \
      \( \
        \( -type d -a \( -name __pycache__ \) \) \
        -o \
        \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
      \) -exec rm -rf '{}' +

CMD python /igneous/igneous/task_execution.py
