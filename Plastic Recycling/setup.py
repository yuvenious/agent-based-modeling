from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(
	ext_modules = cythonize('recycle_module_cy.pyx'),
	include_dirs=[numpy.get_include()])

setup(
	ext_modules = cythonize('recycle_module_cy_default.pyx'),
	include_dirs=[numpy.get_include()])

"""run commad:
python3 setup.py build_ext --inplace

html file
: cython -a example_cy.pyx
"""

# create a module (example_cy.c)
# then, we can import the module
