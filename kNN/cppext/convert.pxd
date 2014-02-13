#distutils: language = c++
# from libcpp.vector cimport vector
# from unordered_map cimport unordered_map
# from unordered_set cimport unordered_set
from libcpp.vector cimport vector
from kNN.cppext.container cimport unordered_map, unordered_set

cdef vector[unordered_map[int, double]] cythonize_X(object)
cdef vector[vector[int]] cythonize_Y(object)
cdef unordered_map[int, unordered_set[int]] cythonize_index(object)
cdef unordered_map[int,int] cythonize_counter(object)