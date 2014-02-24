#distutils: language = c++
from lshtc4.container cimport unodered_set, unordered_map
from lshtc4.utils cimport ModdedWord
from libcpp.vector cimport vector
# cimport numpy as np

# ctypedef np.uint32_t uint
# ctypedef np.float32_t flt
from lshtc4.utils cimport uint, flt

cdef void get_cossim(ModdedWord[:]& d_i, object& t_X, size_t k,
	unordered_map[uint, unordered_set[uint]]& parents_index,
	unordered_map[uint, unordered_set[uint]]& children_index,
	unordered_map[uint, unordered_set[uint]]& iidx,
	vector[uint]& Xdoc_start_idx, vector[uint]& Xdoc_len_idx,
	unordered_map[uint, vector[flt]]& scores, unordered_map[uint, vector[flt]]& pscores)

cdef void optimized_ranks(unordered_map[uint, vector[flt]]& scores,
	unordered_map[uint, vector[flt]]& pscores, unordered_map[uint, uint]& label_counter,
	flt w1, flt w2, flt w3, flt w4, unordered_map[uint, flt]& ranks)

cdef void predict(unordered_map[uint, flt]& ranks, flt alpha, vector[uint]& predictions)