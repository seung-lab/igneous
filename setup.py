#!/usr/bin/env python
from __future__ import print_function
from distutils.command.build import build
from subprocess import call
import os
import shutil
from setuptools import setup, find_packages, Extension

# NOTE: You must run cython --cplus ./ext/src/third_party/mc/_mesher.pyx before setup
# if _mesher.cpp does not exist.

third_party_dir = './ext/third_party'

setuptools.setup(
    setup_requires=['pbr'],
    pbr=True,
    ext_modules=[
        Extension(
            'igneous._mesher',
            sources=[ os.path.join(third_party_dir, name) for name in ('mc/_mesher.cpp','mc/cMesher.cpp') ],
            depends=[ os.path.join(third_party_dir, 'mc/cMesher.h')],
            language='c++',
            include_dirs=[ os.path.join(third_party_dir, name) for name in ('zi_lib/', 'mc/') ],
            extra_compile_args=[
              '-std=c++11','-O3']) #don't use  '-fvisibility=hidden', python can't see init module
    ],
)

