from distutils.core import setup
from setuptools import find_packages

setup(
    name='MuZero',
    version='0.2',
    packages=find_packages(),
    install_requires=['gym', 'tensorflow==2.5.1', 'numpy'],
    license='MIT License',
    author='Radi Achkik',
    author_email='radi.achkik@gmail.com',
    description='A minimal implementation of the MuZero algorithm, based on the paper and pseudocode released by DeepMind')
