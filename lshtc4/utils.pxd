#distutils: language = c++
ctypedef unsigned int uint

cdef extern from "<algorithm>" namespace "std":
    void partial_sort_1 "std::partial_sort"[RandomAccessIterator](RandomAccessIterator first,\
    	RandomAccessIterator middle, RandomAccessIterator last)
    void partial_sort_2 "std::partial_sort"[RandomAccessIterator, Compare]\
    	(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last,\
    		Compare comp)

cdef struct Word:
	uint doc, word, count

cdef struct ModdedWord:
	uint doc, word
	float tfidf

cdef struct Label:
	uint doc, label