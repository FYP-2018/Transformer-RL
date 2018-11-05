#nsml: tensorflow/tensorflow:latest-gpu-py3

from distutils.core import setup
setup(name='nsml-transformer-summarization',
      version='1.0',
      install_requires=[
        'cloudpickle',
        'protobuf>=3.6.0',
        'numpy>=1.13.0',
        'h5py>=2.8.0',
        'nltk>=3.2.4',
        'regex>=2017.6.7',
        'tqdm'])

