#!/usr/bin/env python
import setuptools

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  entry_points={
    "console_scripts": [
      "igneous=igneous_cli:main"
    ],
  },
  pbr=True,
)

