from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
INSTALL_REQUIRES = [
    'ldpc',
    'bposd',
    'networkx',
]

setup(
    name='qmemory_simulation',
    version='0.1',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)