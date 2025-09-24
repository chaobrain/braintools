# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages, setup

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'braintools/', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    README = f.read()

# installation packages
packages = find_packages(
    exclude=[
        "docs*",
        "dev*",
        "tests*",
        "examples*",
        "build*",
        "dist*",
        "braintools.egg-info*",
        "braintools/__pycache__*"
    ]
)

all_dependencies = ['matplotlib', 'nevergrad', 'scipy', 'msgpack']

# setup
setup(
    name='braintools',
    version=version,
    description='Modeling tools for brain simulation.',
    long_description=README,
    long_description_content_type="text/markdown",
    author='BrainTools Developers',
    author_email='chao.brain@qq.com',
    packages=packages,
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.15',
        'brainstate>=0.1.0',
        'brainunit>=0.0.8',
        'typing_extensions',
    ],
    url='https://github.com/chaobrain/braintools',
    project_urls={
        "Bug Tracker": "https://github.com/chaobrain/braintools/issues",
        "Documentation": "https://braintools.readthedocs.io/",
        "Source Code": "https://github.com/chaobrain/braintools",
    },
    extras_require={
        'cpu': ['jax[cpu]', 'brainstate[cpu]', 'brainunit[cpu]'] + all_dependencies,
        'cuda12': ['jax[cuda12]', 'brainstate[cuda12]', 'brainunit[cuda12]'] + all_dependencies,
        'tpu': ['jax[tpu]', 'brainstate[tpu]', 'brainunit[tpu]'] + all_dependencies,
    },
    keywords=(
        'computational neuroscience, '
        'brain-inspired computation, '
        'brain dynamics programming'
    ),
    classifiers=[
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache-2.0 license',
)
