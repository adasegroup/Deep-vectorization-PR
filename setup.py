#!/usr/bin/env python

from setuptools import setup, find_packages
# thanks rakhimovv
__version__ = '0.1.0'
url = 'https://github.com/adasegroup/Deep-vectorization-PR'
install_requires = [
    'hydra-core',
    'pytorch-lightning'
]

setup(name='dpvectpr',
      version=__version__,
      description='Deep_vectorization',
      author='3ddl',
      author_email='',
      url=url,
      install_requires=install_requires,
      packages=find_packages()
      )