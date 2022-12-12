from distutils.core import setup
from pathlib import Path

import setuptools

def get_install_requires() -> [str]:
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets

setup(name='yolos_thermal_tracking',
    version='1.0',
    description='This is a approach for tracking people using thermal imaging cameras.',
    author='Ilya Zaychuk',
    author_email='zajchuk.ilya@gmail.com',
    url='https://github.com/GhostLate/MultiVisual-Dash',
    packages=setuptools.find_packages(),
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=get_install_requires(),
)