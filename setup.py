from setuptools import setup, find_packages

setup(
    name='cyclopeps',
    version='0.1',
    author='Phillip Helms',
    author_email='phelms@caltech.edu',
    url='https://github.com/philliphelms/cyclopeps/',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'mpi4py',
        'psutil',
        'scipy',
    ],
)
