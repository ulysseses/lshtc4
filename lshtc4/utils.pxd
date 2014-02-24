#distutils: language = c++
cimport numpy as np
ctypedef np.uint32_t uint
ctypedef np.float32_t flt

cdef extern from "<algorithm>" namespace "std":
    ForwardIterator max_element_1 "std::max_element"[ForwardIterator]\
    	(ForwardIterator first, ForwardIterator last)
    ForwardIterator max_element_2 "std::max_element"[ForwardIterator, Compare]\
    	(ForwardIterator first, ForwardIterator last, Compare comp)
    void partial_sort_1 "std::partial_sort"[RandomAccessIterator](RandomAccessIterator first,\
    	RandomAccessIterator middle, RandomAccessIterator last)
    void partial_sort_2 "std::partial_sort"[RandomAccessIterator, Compare]\
    	(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last,\
    		Compare comp)

cdef struct Word:
	uint doc, word, count

cdef struct ModdedWord:
	uint doc, word
	flt tfidf

cdef struct Label:
	uint doc, label