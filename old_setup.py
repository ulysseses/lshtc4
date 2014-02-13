from distutils.core import setup, Extension
from Cython.Build import cythonize

''' apparently, you should build clib and tests modules separately '''
#extensions = [ Extension('*', ['clib/*.pyx'], language='c++', include_dirs=['clib/']) ]
#extensions.append([ Extension('*', ['tests/*.pyx'], language='c++', include_dirs=['clib/']) ])
extensions = [ Extension('*', ['tests/*.pyx'], language='c++', include_dirs=['clib/']) ]

setup(
	#ext_modules = cythonize("tests/cython_benchmark.pyx",
	#    language="c++",)
	ext_modules = cythonize(extensions),
	)