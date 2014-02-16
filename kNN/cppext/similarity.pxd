#distutils: language = c++
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from kNN.cppext.container cimport unordered_map, unordered_set

cdef void transform_tfidf(vector[unordered_map[int,double]]&,
	unordered_map[int, int]&)
# cdef double norm(unordered_map[int, double]&)
cdef pair[unordered_map[int, vector[double]], unordered_map[int, vector[double]]] cossim(unordered_map[int, double]&, vector[unordered_map[int,double]]&, int, vector[vector[int]]&, unordered_map[int, unordered_set[int]]&, unordered_map[int, unordered_set[int]]&)
cdef pair[unordered_map[int, vector[double]], unordered_map[int, vector[double]]] cossim2(unordered_map[int, double]&, vector[unordered_map[int,double]]&, int, vector[vector[int]]&, unordered_map[int, unordered_set[int]]&, unordered_map[int, unordered_set[int]]&, unordered_map[int, unordered_set[int]]&)
# cdef double custom_max(unordered_map[int, double])
# cdef double csum(vector[double]&)
cdef unordered_map[int, double] optimized_ranks(unordered_map[int, vector[double]]&, unordered_map[int, vector[double]]&, unordered_map[int, int]&, double, double, double, double)
cdef vector[int] predict(unordered_map[int, double]&, double)