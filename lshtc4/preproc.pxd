#distutils: language = c++
from lshtc4.container cimport unordered_map, unordered_set
from libcpp.vector cimport vector
# cimport numpy as np

# ctypedef np.uint32_t uint
from lshtc4.utils cimport uint, flt

cdef void extract_parents(object Y, infilename, \
    unordered_map[uint, unordered_set[uint]]& parents_index)
cdef void parents2children(unordered_map[uint, unordered_set[uint]]& parents_index, \
    unordered_map[uint, unordered_set[uint]]& children_index)
cdef void inverse_index(object X, unordered_map[uint, unordered_set[uint]]& iidx)
cdef void get_doc_lens(object corpus, vector[uint]& doc_len_idx)
cdef void get_doc_starts(vector[uint]& doc_len_idx, vector[uint]& doc_start_idx)
cdef void doc2label(object Y, unordered_map[uint, unordered_set[uint]]& label_index)