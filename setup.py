#!/usr/bin/env python
import setuptools

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  entry_points={
    "console_scripts": [
      "igneous=igneous_cli:main"
    ],
  },
  extras_require={
    "mysql": [
      "mysql-connector-python",
    ],
  },
  long_description_content_type="text/markdown",
  pbr=True,
)

