#distutils: language = c++
from libcpp.vector import vector
from libcpp.utility cimport pair
from unordered_map import unordered_map
from unordered_set import unordered_set

ctypedef unordered_map[int, int] iidict
ctypedef unordered_map[int, vector[double]] vectmap
ctypedef unordered_map[int, unordered_set[int]] isdict

cdef void transform_tfidf(mapvect&, iidict&)
cdef double norm(iddict&)
cdef pair[vectmap, vectmap] cossim(iddict&, mapvect&, int, vectvect&, isdict&,
	isdict&)
cdef double custom_max(iddict)
cdef double csum(vector[double]&)
cdef iddict optimized_ranks(vectmap&, vectmap&, iidict&,
    double, double, double, double)
cdef vector[int] predict(iddict&, double)