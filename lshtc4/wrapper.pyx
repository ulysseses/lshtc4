#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
'''
Limited Functionality Wrapper to "wrap" C++ customized STL containers
for ease of pickling and unpickling. Pickling works only with
Pythonic objects, so devise a mechanism to pack content into the
STL container when loading a pickle, and to unpack content from the STL
container when dumping into a pickle.

Note: `variable = None` may seem un-necessary, but it is for explicit
ensuring that CPython's garbage collection does indeed do its job.
'''
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, \
    postincrement as inc2
from lshtc4.container cimport unordered_map, unordered_set
# cimport numpy as np

import cPickle as pickle

# ctypedef np.uint32_t uint
from lshtc4.utils cimport uint, flt

cdef void pickleIndex(unordered_map[uint, unordered_set[uint]]& input, char* outname):
	''' pickle Index in a way that it can be pickled '''
	lst = []
	cdef pair[uint, unordered_set[uint]] x
	cdef uint y
	for x in input:
		# convert STL to Py set
		s = set([])
		for y in x.second:
			s.add(y)
		lst.append((x.first, s))
	with open(outname, 'rb') as picklefile:
		pickler = pickle.Pickler(picklefile, -1)
		pickler.dump(lst)
		pickler.clear_memo()
		lst = None
		del pickler
	del lst

cdef void unpickleIndex(unordered_map[uint, unordered_set[uint]]& output, char* inname):
	''' unpickle Pythonized Index into STL Index `output` '''
	with open(inname, 'rb') as picklefile:
		lst = pickle.load(picklefile)
	cdef uint f
	cdet uint x
	cdef unordered_set[uint] cs
	for f,s in lst:
		# convert Py set to STL set
		for x in s:
			cs.insert(x)
		output[f] = cs
		cs.clear()
	lst = None

cdef void pickleVector(vector[uint]& input, char* outname):
	lst = []
	cdef uint x
	for x in input:
		lst.append(x)
	with open(outname, 'wb') as picklefile:
		pickler = pickle.Pickler(picklefile, -1)
		pickler.dump(lst)
		pickler.clear_memo()
		lst = None
		del pickler
	del lst

cdef void unpickleVector(vector[uint]& output, char* inname):
	with open(inname, 'rb') as picklefile:
		lst = pickle.load(picklefile)
	cdef uint x
	for x in lst:
		output.push_back(x)
	lst = None

