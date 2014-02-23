#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
from __future__ import division
import tables as tb
from libc.math cimport log
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc
from lshtc4.utils cimport ModdedWord, partial_sort_2
import numpy as np
cimport numpy as np
from lshtc4 cimport sparse
from lshtc4 import sparse

ctypedef np.uint32_t uint
ctypedef np.float32_t flt

ctypedef bint (*Compare)(pair[uint, flt], pair[uint, flt])
cdef inline bint comp_pair(pair[uint, flt] x, pair[uint, flt] y):
    ''' A comparison func. that returns 1/True or 0/False if x > y
    based on the value of the second element in the pair, respectively. '''
    return <bint> (x.second > y.second)


'''
PARAMS:
cdef ModdedWord[:] d_i
cdef object t_X # tb.table.Table
cdef object t_Y # tb.table.Table
cdef unordered_map[uint, unordered_set[uint]] parents_index
cdef unordered_map[uint, unordered_set[uint]] children_index
cdef unordered_set[uint] iidx
cdef vector[uint] doc_len_idx
cdef vector[uint] doc_start_idx

LOCAL:
cdef set[uint] doc_nums
cdef unordered_map[uint, uint] ref # row_ind -> doc_id "reference"
cdef flt[:] M_data
cdef uint[:] M_indices
cdef uint[:] M_indptr
cdef flt[:] output_vector # partial_sort with NumPy v1.8; use `ref` to see
	corresponding doc_id to score
'''
cdef pair[unordered_map[uint, vector[flt]], unordered_map[uint, vector[flt]]] \
	cossim(ModdedWord[:]& d_i, object& t_X, size_t k, object& t_Y,
	unordered_map[uint, unordered_set[uint]]& parents_index,
	unordered_map[uint, unordered_set[uint]]& children_index,
	vector[uint]& doc_len_idx, vector[uint]& doc_start_idx):
	
	pass







