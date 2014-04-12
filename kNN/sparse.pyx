#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from types cimport *

import numpy as np
cimport numpy as np

cdef void cooBT2csrMV(object& X, cset[uint]& doc_nums, vector[uint]& ref,
	flt[:]& M_data, uint[:]& M_indices, uint[:]& M_indptr, flt[:]& M_norm):
	''' Coordinate-list (COO) BLZ table (BT) -> compressed sparse row (CSR) Cython-memoryviews (MV)
		containing `data`, `indices`, and `indptr`. In addition to these memoryviews, the vector
		parameter is modified to keep track of which row corresponds to which doc_id. Also in addition, 
		this function calculates the norm of each doc and returns `M_norm` within the tuple. '''
	num_words = X.num_words
	num_docs = len(X)
	M_data = np.empty(num_words, dtype=np.float32)
	M_norm = np.empty(len(X))
	M_indices = np.empty(num_words, dtype=np.uint32)
	M_indptr = np.empty(num_words+1, dtype=np.uint32)
	M_indptr[num_rows] = num_rows + 1 #<-- HACK

	cdef size_t i_M_data, i_M_indices, i_M_indptr, first_ind
	cdef size_t i, j, first_ind
	cdef Word[:] sa_mv
	cdef Word word_obj
	i_M_data, i_M_indices, i_M_indptr, first_ind = 0, 0, 0, 0

	# Iterate each doc in `X` and push into respective space within
	# `data`, `indices`, and `indptr`
	ref.resize(num_docs)
	for i,doc_num in enumerate(doc_nums):
		# get the next training doc indexed by i
		sa_mv = X[doc_num]
		first_ind = i_M_data
		for j in xrange(len(sa_mv)):
			word_obj = sa_mv[j]
			M_data[i_M_data] = word_obj.tfidf
			M_norm[i] += word_obj.tfidf ** 2
			i_M_data += 1
			M_indices[i_M_indices] = word_obj.word
			i_M_indices += 1
		M_indptr[i_M_indptr] = first_ind
		M_norm[i] = M_norm[i] ** 0.5
		ref[i] = doc_num
		i_M_indptr += 1


cdef void coo2csr(Word[:]& in_v, flt[:]& out_v_data, size_t[:]& out_v_indices):
	''' Convert COO Word[:] MV to data and indices memoryviews. '''
	out_v_data = in_v[2,:]
	out_v_indices = in_v[1,:]


cdef void sp_Mv_mult(flt[:]& M_data, uint[:]& M_indices, uint[:]& M_indptrs,
		flt[:]& v_data, uint[:]& v_indices, vector[uint]& ref,
		DSPair[:]& output):
	''' Multiplies sparse matrix `M` with sparse vector `v`. '''
	# Memoryviews are curiously returned by reference
	# Rename M_indptr to M_indptrs b/c it makes more sense to my mind
	cdef:
		uint M_indptr0 = 0
		size_t i
		uint M_indptr1
		flt[:] data
		uint[:] cols
		uint cols_len, vi_len
	output = np.empty(len(v_data), dtype='u4,f4') # <-- maybe work? check!
	#M_indptrs[len(M_indptrs)-1] += M_indptrs[len(M_indptrs)-2] #<-- HACK
	for i in xrange(len(M_indptrs) - 1): #<-- HACK
		# Select the corresponding cols & data
		M_indptr1 = M_indptrs[i+1]
		data = M_data[M_indptr0 : M_indptr1]
		cols = M_cols[M_indptr0 : M_indptr1]
		M_indptr0 = M_indptr1
		# Dot product: iterate over the smaller indices of the two
		cols_len, vi_len = len(cols), len(v_indices)
		if cols_len < vi_len:
			output[i] = DSPair(ref[i], sp_uv_dot(data, cols,
				v_data, v_indices)) # outputs a (doc, score) pair
		else:
			output[i] = DSPair(ref[i], sp_uv_dot(v_data, v_indices,
				data, cols)) # outputs a (doc, score) pair


cdef flt sp_uv_dot(flt[:]& u_data, uint[:]& u_indices, flt[:]& v_data,
		uint[:]& v_indices): # static (exists only in this module)
	''' Multiplies sparse vector `u` with sparse vector `v` '''
	# Assume len(u) <= len(v)
	cdef:
		uint len_u = len(u)
		size_t i_u = 0
		size_t i_v = 0
		flt ansatz = 0
	while i_u < len_u:
		if u_indices[i_u] > v_indices[i_v]:
			i_v += 1
		elif u_indices[i_u] < v_indices[i_v]:
			i_u += 1
		elif u_indices[i_u] == v_indices[i_v]:
			ansatz += u_data[i_u] * v_data[i_v]
			i_v += 1
			i_u += 1
	return ansatz
	