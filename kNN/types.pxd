#distutils: language = c++
from libcpp.vector cimport vector
from container cimport unordered_map, unordered_set
cimport numpy as np


ctypedef np.uint32_t uint
ctypedef np.float32_t flt

cdef struct Word:
	uint doc
	uint word
	float tfidf

cdef struct Label:
	uint doc
	uint label

cdef struct Doc: # instead of `Word`, it's inverted: `Doc`
	uint word
	uint doc

cdef struct DSPair: # DSPair stands for doc-score pair
	uint doc
	flt score

ctypedef unordered_map[uint, unordered_set[uint]] Family
ctypedef unordered_map[uint, unordered_set[uint]].iterator Family_it
ctypedef unordered_map[uint, vector[flt]] Score
ctypedef unordered_map[uint, vector[flt]].iterator Score_it
