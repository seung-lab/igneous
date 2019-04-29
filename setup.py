#!/usr/bin/env python
from __future__ import print_function
from distutils.command.build import build
from subprocess import call
import os
import shutil
import setuptools
import numpy as np

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
     ':python_version == "2.7"': ['futures'],
     ':python_version == "2.6"': ['futures'],
  },
  pbr=True,
  ext_modules=[],
)

