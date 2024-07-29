from setuptools import setup, find_packages

# read the contents of your README file
from os import path
with open(('README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()
    
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'matplotlib',
    'ldpc',
    'bposd',
]

setup(
    name='aperiodic_codes',
    version='0.1',
    description='Constructing classical and quantum error corrections defined on aperiodic tilings.',
    long_description=long_description,
    author='Yi Tang',
    author_email='rosetta_tan@outlook.com',
    packages=find_packages(include=['aperiodic_codes', 'aperiodic_codes.*']),
    install_requires=INSTALL_REQUIRES
)