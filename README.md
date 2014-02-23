###TODO

	cdef unordered_map[uint, unordered_set[uint]] parents_index
	cdef unordered_map[uint, unordered_set[uint]] children_index
	cdef unordered_map[uint, unordered_set[uint]] iidx
	cdef vector[uint] doc_len_idx
	cdef vector[uint] doc_start_idx

* Complete `cossim` within `similarity.pyx`
* Unit tests!
* Implement an additional locality sensitive hashing (LSH) based NN-search