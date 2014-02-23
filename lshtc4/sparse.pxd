#distutils: language = c++
import numpy as np
cimport numpy as np

ctypedef np.int32_t uint
ctypedef np.float32_t flt

cdef flt sp_uv_dot(flt[:]& u_data, uint[:]& u_indices, flt[:]& v_data,
		uint[:]& v_indices)

cdef flt[:] sp_Mv_mult(flt[:]& M_data, uint[:]& M_indices,
		uint[:]& M_indptrs, flt[:]& v_data, uint[:]& v_indices)

