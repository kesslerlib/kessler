import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'kessler'
MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
setup(
    name='kessler',
    version=read_package_variable('__version__'),
    description='Simulation-based inference and machine learning for space collision assessment and avoidance.',
    author='ESA FDL Europe Constellations Team',
    # author_email='',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'torch>=1.5.1', 'pykep>=2.5', 'skyfield>=1.26', 'kessler', 'pandas'],
    extras_require={'dev': ['pytest', 'pytest-xdist'],
                    'docs': ['sphinx==3.2.1', 'sphinx_rtd_theme==0.5.2', 'jupyter-sphinx==0.3.2', 'myst-nb==0.12.3']},
    # url='https://github.com/kessler/kessler',
    classifiers=['License :: OSI Approved :: BSD License', 'Programming Language :: Python :: 3'],
    license='BSD'
    # keywords='',
)
