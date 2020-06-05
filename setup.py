import sys
from distutils.core import setup

if sys.version_info < (3, 6, 3):
    print('This package requires Python >= 3.6.3')
    sys.exit(1)

setup(
    name='mbae',
    version='0.1dev1',
    description='',
    author='Ilia Korvigo',
    author_email='',
    url='',
    packages=['mbae', 'mbae_resources'],
    package_data={
        'mbae_resources': [
            'alleles.faa',
            'consurf.tsv',
            'encoder.joblib',
            'config.json',
            '*.h5'
        ]
    },
    install_requires=[
        "importlib_resources ; python_version<'3.7'",
    ]
)
