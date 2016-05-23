from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = '1'  # use '' for first of series, number for 1 and above
#_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "Pattern search optimization dealing intelligently with noisy function evaluation" 
# Long description will go up on the pypi page
long_description = """
Noisyopt is a library for local optimization of noisy objective functions.

Optimization in the presence of stochasticity in the objective function
evaluation is challenging. This library provides a simple, robust algorithm
to solve this problem based on pattern search with adaptive sampling of
the noisy objective function.

For more info see the `documentation <http://noisyopt.readthedocs.io/en/latest/>`_ or the `source code <http://github.com/andim/noisyopt>`_.
"""

NAME = "noisyopt"
MAINTAINER = "Andreas Mayer"
MAINTAINER_EMAIL = "andisspam@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/andim/noisyopt"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Andreas Mayer"
AUTHOR_EMAIL = "andisspam@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['noisyopt',
            'noisyopt.tests']
PACKAGE_DATA = {}
REQUIRES = ["numpy", "scipy"]
