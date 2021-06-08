#!/usr/bin/env python
import setuptools

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  entry_points={
    "console_scripts": [
      "igneous=igneous_cli:main"
    ],
  },
  long_description_content_type="text/markdown",
  pbr=True,
)

