#distutils: language = c++
from libcpp.vector cimport vector
from container cimport unordered_map, unordered_set
cimport numpy as np


'''
-------------------
---- TYPES.HPP ----
-------------------

typedef float flt;
typedef size_t DOC;
typedef size_t WORD;
typedef size_t LABEL;
typedef size_t CAT;
typedef size_t PCAT;

struct Word {
	size_t doc, word
	float tfidf
};

struct Label {
	size_t doc, label
};

struct Doc {
	size_t word, doc
};
'''


# cdef extern from "types.hpp":
# 	ctypedef float flt
# 	ctypedef size_t DOC
# 	ctypedef size_t WORD
# 	ctypedef size_t LABEL
# 	ctypedef size_t CAT
# 	ctypedef size_t PCAT

# 	cdef struct Word:
# 		size_t doc
# 		size_t word
# 		float tfidf

# 	cdef struct Label:
# 		size_t doc
# 		size_t label

# 	cdef struct Doc:
# 		size_t word
# 		size_t doc


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
