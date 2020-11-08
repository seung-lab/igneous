#!/usr/bin/env python
import setuptools

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
    "deflate": [ "deflate" ],
  },
  pbr=True,
)

