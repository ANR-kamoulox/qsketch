# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

import shutil
if shutil.which('nvcc'):
    # CUDA is there.
    torch_packages = [
            "torch>=1.4.0",
            "torchvision>=0.5.0"
        ]
else:
    torch_packages = [
            "torch>=1.4.0[cpu]",
            "torchvision>=0.5.0[cpu]"
        ]

# Proceed to setup
setup(
    name='qsketch',
    version='0.3',
    description='Tools for Sliced Wasserstein and friends',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/aliutkus/qsketch',
    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    keywords='compressed learning sliced wasserstein',
    install_requires=[
        *torch_packages,
        'torchsearchsorted @ git+https://github.com/aliutkus/torchsearchsorted',
        'torchpercentile @git+https://github.com/aliutkus/torchpercentile',
        'torchpercentile @git+https://github.com/aliutkus/torchinterp1d',
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ]
    )
