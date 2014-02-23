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

ctypedef np.int32_t uint
ctypedef np.float32_t flt

cdef flt sp_uv_dot(flt[:]& u_data, uint[:]& u_indices, flt[:]& v_data,
		uint[:]& v_indices):
	''' assume len(u) <= len(v) '''
	cdef:
		size_t len_u = len(u)
		size_t i_u = 0
		size_t i_v = 0
		flt ansatz = 0
		uint u_col, v_col
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

cdef flt[:] sp_Mv_mult(flt[:]& M_data, uint[:]& M_indices,
		uint[:]& M_indptrs, flt[:]& v_data, uint[:]& v_indices):
	# Memoryviews are curiously returned by reference
	# Renamed M_indptr to M_indptrs b/c it makes more sense to my mind
	cdef:
		flt[:] output_vector = np.empty(len(M_indptrs) - 1, dtype=np.float32)
		uint M_indptr0 = 0
		size_t i
		uint M_indptr1
		flt[:] data
		uint[:] cols
		flt ansatz
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

def cooTable2cscMV(object& table, set[uint]& doc_nums, 
		vector[uint]& doc_start_idx, vector[uint]& doc_len_idx, 
		unordered_map[uint, uint]& ref):
	''' Coordinate-list (COO) table (T) -> 3 Compressed sparse
		column (CSC) cython-memoryviews containing `data`, `indices`,
		and `indptr`. In addition to these memoryviews, the unordered_map
		parameter is modified to keep track of which row corresponds to which
		doc_id.

	    Returned as a tuple (object) because memoryviews are also PyObjects!
	'''
	cdef uint num_words = 0
	cdef size_t i
	for i in xrange(doc_nums.size()):
		num_words += doc_len_idx[doc_nums[i]]
	cdef flt[:] M_data = np.empty(num_words, dtype=np.float32)
	cdef uint[:] M_indices = np.empty(num_words, dtype=np.uint32)
	cdef uint[:] M_indptr = np.empty(num_words+1, dtype=np.uint32)
	M_indptr[num_rows] = num_rows + 1 #<-- HACK

	cdef uint doc_num, doc_len, doc_start
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


