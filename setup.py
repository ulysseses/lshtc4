# build script for `lshtc4`
import sys, os
from distutils.core import setup
from distutils.extension import Extension
import numpy

# We'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print "You don't seem to have Cython installed. Please get a"
    print "copy from www.cython.org and install it"
    sys.exit(1)


def scandir(dir, files=[]):
    ''' Scan the `dir` directory recursively for extension files, converting
        them to extension names in dotted notation '''
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, '.')[:-4]) # :-4 removes .pyx
        elif os.path.isdir(path):
            scandir(path, files)
    # fix minor issue when there are files whose first 1 or 2 chars are dots
    for i in xrange(len(files)):
        file = files[i]
        while file[0] == '.':
            file = file[1:]
        files[i] = file
    return files


def makeExtension(extName):
    ''' Generate an Extension object from its dotted name '''
    extPath = extName.replace('.', os.path.sep) + ".pyx"
    return Extension(extName,
                     [extPath],
                     include_dirs       = ['.'],
                     language           = "c++",
                     extra_compile_args = [],
                     extra_link_args    = [],
                     libraries          = []
                    )


# Get the list of extensions
extNames = scandir("kNN/")
# Build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]
# Finally, we can pass all this to distutils
setup(name= "kNN",
      packages= ["kNN"],
      ext_modules= extensions,
      include_dirs= [numpy.get_include()],
      cmdclass= {"build_ext": build_ext}
     )




