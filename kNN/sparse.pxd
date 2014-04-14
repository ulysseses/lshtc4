#distutils: language = c++
from libcpp.set cimport set as cset
from libcpp.utility cimport pair
from types cimport * # <-- hope this works
from preproc cimport Baggage

cdef void cooBT2csrMV(Baggage& X, cset[uint]& doc_nums, vector[uint]& ref,
	flt[:]& M_data, uint[:]& M_indices, uint[:]& M_indptr, flt[:]& M_norm)

cdef void coo2csr(Word[:]& in_v, flt[:]& out_v_data, size_t[:]& out_v_indices)

cdef void sp_Mv_mult(flt[:]& M_data, uint[:]& M_indices, uint[:]& M_indptrs,
	flt[:]& v_data, uint[:]& v_indices, vector[uint]& ref,
	DSPair[:]& output)

