import sys
from distutils.core import setup

from setuptools import find_packages

if sys.version_info < (3, 7, 0):
    print('This package requires Python >= 3.7.0')
    sys.exit(1)

setup(
    name='mbae',
    version='0.1dev1',
    description='',
    author='Ilia Korvigo',
    author_email='',
    url='',
    packages=find_packages(),
    package_data={
        'mbae_resources': [
            'binding_regions.fsa',
            'consurf.tsv',
            'config.json',
            '*.h5'
        ]
    },
    install_requires=[
        "importlib_resources ; python_version<'3.7'",
        'click>=7.1.2',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'biopython>=1.19',
        'toolz>=0.10.0',
        'tensorflow==2.2',
        'wget>=3.2',
        'fn',
        'tqdm'
    ],
    scripts=['mbae.py']
)
