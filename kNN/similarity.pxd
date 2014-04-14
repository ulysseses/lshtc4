#distutils: language = c++
from libcpp.vector cimport vector
from container cimport unordered_map

from types cimport * # <-- hope this works!
from preproc cimport Baggage

cdef void cossim(Word[:]& d_i, Baggage& t_X, size_t k, Family& parents_index,
	Family& children_index, Baggage& labels_index, Baggage& iidx,
	Score& scores, Score& pscores)

cdef void ranks(Score& scores, Score& pscores, unordered_map[size_t,size_t]& label_counter,
	flt w1, flt w2, flt w3, flt w4, unordered_map[size_t,flt]& ranks)

cdef void predict(unordered_map[uint, flt]& ranks, flt alpha, vector[uint]& predictions)