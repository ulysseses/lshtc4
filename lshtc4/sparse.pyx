#cython: boundscheck=False
#cython: wraparound=False
"""
def sp_uv_dot(u_data, u_indices, v_data, v_indices):
	''' assume len(u) <= len(v) '''
	len_u = len(u)
	i_u, i_v = 0, 0
	ansatz = 0
	while i_u < len_u:
		u_col, v_col = u_indices[i_u], v_indices[i_v]
		if u_indices[i_u] > v_indices[i_v]:
			i_v += 1
		elif u_indices[i_u] == v_indices[i_v]:
			ansatz += u_data[i_u] * v_data[i_v]
			i_u += 1; i_v += 1
		else:
			i_u += 1
	return ansatz

def sp_Mv_mult(M_data, M_indices, M_indptrs, v_data, v_indices):
	# Renamed M_indptr to M_indptrs b/c it makes more sense to my mind
	output_vector = np.empty(len(M_indptrs) - 1, dtype=np.float32)
	M_indptr0 = 0
	for i in xrange(len(M_indptrs) - 1): #<-- hackity hack
		# Select the corresponding cols & data
		M_indptr1 = M_indptrs[i+1]
		data = M_data[M_indptr0 : M_indptr1]
		cols = M_cols[M_indptr0 : M_indptr1]
		M_indptr0 = M_indptr1
		# Dot product: iterate over the smaller indices of the two
		ansatz = 0
		cols_len, vi_len = len(cols), len(v_indices)
		if cols_len < vi_len:
			output_vector[i] = sp_uv_dot(data, cols, v_data, v_indices)
		else:
			output_vector[i] = sp_uv_dot(v_data, v_indices, data, cols)
		M_indptr0 = M_indptr1
	return output_vector
"""
import numpy as np
cimport numpy as np
cimport cython
from libcpp.set cimport set
from libcpp.vector cimport vector
from kNN.cppext.container cimport unordered_map
from lshtc4.utils cimport ModdedWord

np.import_array()

UINT = np.uint32
FLT = np.float32

ctypedef np.int32_t UINT_t
ctypedef np.float32_t FLT_t

cdef FLT_t sp_uv_dot(FLT_t[:]& u_data, UINT_t[:]& u_indices, FLT_t[:]& v_data,
		UINT_t[:]& v_indices):
	''' assume len(u) <= len(v) '''
	cdef:
		size_t len_u = len(u)
		size_t i_u = 0
		size_t i_v = 0
		FLT_t ansatz = 0
		UINT_t u_col, v_col
	while i_u < len_u:
		u_col, v_col = u_indices[i_u], v_indices[i_v]
		if u_indices[i_u] > v_indices[i_v]:
			i_v += 1
		elif u_indices[i_u] == v_indices[i_v]:
			ansatz += u_data[i_u] * v_data[i_v]
			i_u += 1; i_v += 1
		else:
			i_u += 1
	return ansatz

cdef FLT_t[:] sp_Mv_mult(FLT_t[:]& M_data, UINT_t[:]& M_indices,
		UINT_t[:]& M_indptrs, FLT_t[:]& v_data, UINT_t[:]& v_indices):
	# Memoryviews are curiously returned by reference
	# Renamed M_indptr to M_indptrs b/c it makes more sense to my mind
	cdef:
		FLT_t[:] output_vector = np.empty(len(M_indptrs) - 1, dtype=FLT)
		UINT_t M_indptr0 = 0
		size_t i
		UINT_t M_indptr1
		FLT_t[:] data
		UINT_t[:] cols
		FLT_t ansatz
		size_t cols_len, vi_len
	#M_indptrs[len(M_indptrs)-1] += M_indptrs[len(M_indptrs)-2] #<-- HACK
	for i in xrange(len(M_indptrs) - 1): #<-- HACK
		# Select the corresponding cols & data
		M_indptr1 = M_indptrs[i+1]
		data = M_data[M_indptr0 : M_indptr1]
		cols = M_cols[M_indptr0 : M_indptr1]
		M_indptr0 = M_indptr1
		# Dot product: iterate over the smaller indices of the two
		ansatz = 0
		cols_len, vi_len = len(cols), len(v_indices)
		if cols_len < vi_len:
			output_vector[i] = sp_uv_dot(data, cols, v_data, v_indices)
		else:
			output_vector[i] = sp_uv_dot(v_data, v_indices, data, cols)
		M_indptr0 = M_indptr1
	return output_vector

def cooTable2cscMV(object& table, set[UINT_t]& doc_nums, 
		vector[UINT_t]& doc_start_idx, vector[UINT_t]& doc_len_idx, 
		unordered_map[UINT_t, UINT_t]& ref):
	''' Coordinate-list (COO) table (T) -> 3 Compressed sparse
		column (CSC) cython-memoryviews containing `data`, `indices`,
		and `indptr`. In addition to these memoryviews, the unordered_map
		parameter is modified to keep track of which row corresponds to which
		doc_id.

	    Returned as a tuple (object) because memoryviews are also PyObjects!
	'''
	cdef UINT_t num_words = 0
	cdef size_t i
	for i in xrange(doc_nums.size()):
		num_words += doc_len_idx[doc_nums[i]]
	cdef FLT_t[:] M_data = np.empty(num_words, dtype=FLT)
	cdef UINT_t[:] M_indices = np.empty(num_words, dtype=UINT)
	cdef UINT_t[:] M_indptr = np.empty(num_words+1, dtype=UINT)
	M_indptr[num_rows] = num_rows + 1 #<-- HACK

	cdef UINT_t doc_num, doc_len, doc_start
	cdef ModdedWord[:] sa_mv
	cdef ModdedWord word_obj
	cdef size_t j
	cdef i_M_data, i_M_indices, i_M_indptr, first_ind
	i_M_data, i_M_indices, i_M_indptr, first_ind = 0, 0, 0, 0
	# Iterate each doc in `table` and push into respective space within
	# `data`, `indices`, and `indptr`
	for i in xrange(doc_nums.size()):
		doc_num = doc_nums[i]
		doc_len = doc_len_idx[doc_num]
		doc_start = doc_start_idx[doc_num]
		sa_mv = table[doc_start : doc_start + doc_len]
		first_ind = i_M_data
		for j in xrange(len(sa_mv)):
			word_obj = sa_mv[i]
			M_data[i_M_data] = word_obj.tfidf
			i_M_data += 1
			M_indices[i_M_indices] = word_obj.word
			i_M_indices += 1
		M_indptr[i_M_indptr] = first_ind
		ref[i_M_indptr] = doc_num
		i_M_indptr += 1
	return (M_data, M_indices, M_indptr)


