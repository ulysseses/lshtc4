# build script for 'lshtc4'
import sys, os
from distutils.core import setup
from distutils.extension import Extension
import numpy

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print "You don't seem to have Cython installed. Please get a"
    print "copy from www.cython.org and install it"
    sys.exit(1)


# scan the 'clib' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, '.')[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    # fix issues where the first 1 or 2 chars are dots...
    for i in xrange(len(files)):
        file = files[i]
        while file[0] == '.':
            file = file[1:]
        files[i] = file
    return files


# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = ["."],   # adding the '.' to include_dirs is CRUCIAL!!
                                             # also, include any .h/.hpp header files
        language='c++',                 # should I generate .cpp or .c source?
        extra_compile_args = ["-O3"],
        #extra_link_args = [],
        #libraries = [],                # put any .o/.so files here
        )

# get the list of extensions
extNames = scandir("lshtc4/")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

# finally, we can pass all this to distutils
setup(
  name="lshtc4",
  packages=["lshtc4", ],
  ext_modules=extensions,
  include_dirs = [numpy.get_include()],
  cmdclass = {'build_ext': build_ext},
)