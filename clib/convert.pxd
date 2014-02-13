#distutils: language = c++
from libcpp.vector import vector
from unordered_map import unordered_map
from unordered_set import unordered_set

cdef vector[unordered_map[int, double]] cythonize_X(object)
cdef vector[vector[int]] cythonize_Y(object)
cdef unordered_map[int, unordered_set[int]] cythonize_index(object)
cdef unordered_map[int,int] cythonize_counter(object)