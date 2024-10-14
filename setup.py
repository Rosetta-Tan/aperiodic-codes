from setuptools import setup, find_packages

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
    description='Constructing classical and quantum error corrections \
        defined on aperiodic tilings.',
    long_description=long_description,
    author='Yi Tan',
    author_email='rosetta_tan@outlook.com',
    packages=find_packages(include=['aperiodic_codes', 'aperiodic_codes.*']),
    package_data={'aperiodic_codes':['cut_and_project/libcnp.so']},
    install_requires=INSTALL_REQUIRES
)
