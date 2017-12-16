#!/usr/bin/env python
from __future__ import print_function
from distutils.command.build import build
from subprocess import call
import os
import shutil
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

def igneous_compile():
    setup_dir = os.path.dirname(__file__)
    src_dir = os.path.join(setup_dir, 'ext/src')
    third_party_dir = os.path.join(setup_dir, 'ext/third_party')

    cythonize(os.path.join(third_party_dir,'mc/_mesher.pyx'))

    setup(
        name='igneous',
        version='0.0.1',
        description='Python data pipeline for neuroglancer Precomputed format',
        author_email='ws9@princeton.edu',
        url='https://github.com/seung-lab/igneous',
        packages=find_packages(),
        package_data={},
        install_requires=[
            "Pillow>=3.2.0",
            "numpy>=1.11.0",
            'requests',
        ],
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
        use_2to3=True,
        cmdclass={}, 
    )

igneous_compile()

