#distutils: language = c++
from libcpp.vector cimport vector
from libcpp.utility cimport pair
# cimport numpy as np

# ctypedef np.int32_t uint
# ctypedef np.float32_t flt
from lshtc4.utils cimport uint, flt

cdef flt sp_uv_dot(flt[:]& u_data, uint[:]& u_indices, flt[:]& v_data,
	uint[:]& v_indices)

cdef void sp_Mv_mult(flt[:]& M_data, uint[:]& M_indices,
	uint[:]& M_indptrs, flt[:]& v_data, uint[:]& v_indices,
	vector[uint]& ref, vector[pair[uint,flt]]& output_vector)

