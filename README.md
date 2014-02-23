###TODO

    cdef set[uint] doc_nums
	cdef unordered_map[uint, unordered_set[uint]] parents_index
	cdef unordered_map[uint, unordered_set[uint]] children_index
	cdef unordered_set[uint] iidx
	cdef vector[uint] doc_len_idx
	cdef vector[uint] doc_start_idx

* Wrap the above containers "Pythonically." We need to do this b/c we want to pickle them. It doesn't have to be advanced, as we're only concerned with their picklability. Once they're pickled, when you want to un-pickle it later, you load it, and then you just get `ContainerWrapper.unwrap()` --> returns C++ STL container. `unwrap` should be a `cdef` function that returns the original C object. We can then dispose of wrapper... or not.
* Complete `cossim` within `similarity.pyx`
* Unit tests!
* Implement an additional locality sensitive hashing (LSH) based NN-search