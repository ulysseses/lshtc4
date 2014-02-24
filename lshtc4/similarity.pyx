#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
from __future__ import division
import tables as tb
from libc.math cimport log
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc, \
	postincrement as inc2
from lshtc4.utils cimport ModdedWord, partial_sort_2, max_element_1
import numpy as np
cimport numpy as np
from lshtc4 cimport sparse
from lshtc4 import sparse

ctypedef np.uint32_t uint
ctypedef np.float32_t flt

ctypedef bint (*Compare)(pair[uint, flt], pair[uint, flt])
cdef inline bint comp_pair(pair[uint, flt] x, pair[uint, flt] y):
    ''' A comparison func. that returns 1/True or 0/False if x > y
    based on the value of the second element in the pair, respectively. '''
    return <bint> (x.second > y.second)

cdef void get_cossim(ModdedWord[:]& d_i, object& t_X, size_t k,
	unordered_map[uint, unordered_set[uint]]& parents_index,
	unordered_map[uint, unordered_set[uint]]& children_index,
	unordered_map[uint, unordered_set[uint]]& iidx,
	vector[uint]& Xdoc_start_idx, vector[uint]& Xdoc_len_idx,
	unordered_map[uint, vector[flt]]& scores, unordered_map[uint, vector[flt]]& pscores):
	# Get candidate docs
	cdef set[uint] doc_nums
	get_candidate_doc_nums(d_i, iidx, doc_nums)
	cdef flt[:] X_data, X_norm
	cdef uint[:] X_indices, X_indptr
	cdef vector[uint] rowID2docID
	X_data, X_indices, X_indptr, X_norm = sparse.cooX2csrMV(t_X, doc_nums,
		Xdoc_start_idx, Xdoc_len_idx, rowID2docID) # rowID2docID is modified here
	# Calculate numerator of cossim
	cdef vector[pair[uint,flt]] doc_scores
	sparse.sp_Xv_mult(X_data, X_indices, X_indptrs, v_data, v_indices, rowID2docID,
		doc_scores)
	# Calculate denominator of cossim
	cdef flt qNorm = 0
	cdef size_t i
	for i in xrange(len(d_i)):
		qNorm += d_i[i].tfidf ** 2
	qNorm = qNorm ** 0.5
	for i in xrange(doc_nums.size()):
		doc_scores[i].second /= qNorm * X_norm[i]
	# Categorize each doc and its respective score
	cdef unordered_map[uint, vector[flt]] cat_scores_dict
	cdef uint label
	for i in xrange(doc_nums.size()):
		label = labels_index[doc_scores[i].first] # doc -> label
		if cat_scores_dict.find(label) == cat_scores_dict.end():
			cat_scores_dict[label] = vector[flt]()
		cat_scores_dict[label].push_back(doc_scores[i].second) # push in score for `label`
	# Return the k-NN (aka top-k similar examples)
	cdef size_t kk
	if k < doc_scores.size():
		kk = k
		partial_sort_2(doc_scores.begin(), doc_scores.begin()+kk, doc_scores.end(),
			comp_pair)
	else:
		kk = doc_scores.size() # don't bother sorting
	# optimized/transformed scores & pscores
	cdef uint label
	cdef unordered_set[uint]* parents_set, children_set
	cdef uint parent
	for i in xrange(kk):
		label = doc_scores[i].first
		if scores.find(label) == scores.end():
			scores[label] = vector[flt]()
		scores[label].push_back(doc_scores[i].second)
		parents_set = &parents_index[label]
		for parent in deref(parents_set):
			children_set = &children_index[parent]
			if pscores.find(label) == pscores.end():
				pscores[label] = vector[flt]()
			pscores[label].push_back(deref(children_set).size())


cdef void get_candidate_doc_nums(ModdedWord[:]& d_i,
	unordered_map[uint, unordered_set[uint]]& iidx,
	set[uint]& doc_nums):
	''' find all candidate docs from the words in `d_i` with `iidx`'''
	cdef uint word
	cdef size_t i
	cdef unordered_map[uint, unordered_set[uint]].iterator got
	cdef unordered_set[uint]* doc_set
	cdef uint doc
	for i in xrange(len(d_i)):
		word = d_i[i].word
		got = iidx.find(word)
		if got != iidx.end():
			doc_set = &(deref(got).second)
			for doc in deref(doc_set):
				doc_nums.insert(doc)

cdef flt csum(vector[flt]& vect):
	''' custom sum func for vector[flt]'s '''
	cdef flt ansatz = 0
	cdef size_t i
	for i in xrange(vect.size()):
		ansatz += vect[i]
	return ansatz

cdef void optimized_ranks(unordered_map[uint, vector[flt]]& scores,
	unordered_map[uint, vector[flt]]& pscores, unordered_map[uint, uint]& label_counter,
	flt w1, flt w2, flt w3, flt w4, unordered_map[uint, flt]& ranks):
	''' w1..w4 are weights corresponding to x1..x4 '''
	cdef pair[uint, vector[flt]] kv
	cdef vector[flt] inner_scores, inner_pscores
	cdef uint c
	cdef flt x1, x2, x3, x4
	for kv in scores:
		c = kv.first
		inner_scores = kv.second
		inner_pscores = pscores[c]
		x1 = deref(max_element_1(inner_scores.begin(), inner_scores.end()))
		x2 = csum(inner_pscores)
		x3 = csum(inner_scores)
		x4 = (<flt>inner_scores.size())/(<flt>label_counter[c])
		ranks[c] = w1*x1 + w2*x2 + w3*x3 + w4*x4

cdef flt custom_max(unordered_map[uint,flt]& lut):
	''' personal func taht returns the max value inside an
		unordered_map[uint, flt] '''
	cdef pair[uint,flt] kv
	cdef flt ansatz = deref(lut.begin()).second
	for kv in lut:
		if kv.second > ansatz:
			ansatz = kv.second
	return ansatz

cdef void predict(unordered_map[uint, flt]& ranks, flt alpha, vector[uint]& predictions):
    ''' Return a vector of labels if their corresponding ranks are higher
        than a threshold provided by `alpha`.
    '''
    cdef flt max_rank = custom_max(ranks)
    cdef pair[uint,flt] kv
    for kv in ranks:
    	if kv.second / max_rank > alpha:
    		ans.push_back(kv.first)







