#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
from __future__ import division

from libc.math cimport log as clog
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.utility cimport pair
from container cimport unordered_map, unordered_set
from cython.operator cimport dereference as deref, preincrement as inc, \
	postincrement as inc2

from types cimport *
from algorithms cimport partial_sort_2, max_element_1

import numpy as np
cimport numpy as np

cimport sparse

ctypedef bint (*Compare)(pair[uint,flt], pair[uint,flt])
cdef inline bint comp_pair(pair[uint,flt] x, pair[uint,flt] y):
	''' A comparison function that returns 1/True or 0/False if x > y
		based on the value of the 2nd element of the pair. '''
	return x.second > y.second

cdef void cossim(Word[:]& d_i, object& t_X, size_t k, Family& parents_index,
	Family& children_index, object& labels_index, object& iidx,
	Score& scores, Score& pscores):
	'''
	Returns the child and parent category scores for the input query document
	based on cosine-similarity with documents in the training set.

	Procedure:

		# COO b-array -> CSR row memoryview
		cooBA2csrMV(doc, tX, iidx) -> spM, spv
		sp_Mv_mult(spM, spv) -> doc_scores
		partial_sort(doc_scores, K, cmp_func)
		categorize(doc_scores, tY) -> scores
		get_pscores(scores, parents_index) -> pscores
		return scores, pscores

	Notes:
		`object`s are technically `Baggage`s.

	'''
	# COO blz.btable -> CSR memview
	cdef:
		cset[uint] doc_nums
		flt[:] X_data, X_norm
		uint[:] X_indices, X_indptr
		vector[uint] rowID2docID
		flt[:]& d_i_data
		uint[:]& d_i_indices
	sparse.coo2csr(d_i, d_i_data, d_i_indices)
	get_candidate_doc_nums(d_i, iidx, doc_nums)
	sparse.cooBT2csrMV(X, doc_nums, rowID2docID,
		X_data, X_indices, X_indptr, X_norm)
	# Calculate the numerator of cossim
	cdef DSPair[:] doc_scores
	sparse.sp_Mv_mult(X_data, X_indices, X_indptr, d_i_data, d_i_indices,
		rowID2docID, doc_scores)
	# Calculate the denominator of cossim
	# note: only X_norm has been calculated, not qNorm yet..
	cdef flt qNorm = 0
	cdef size_t i
	for i in xrange(len(d_i)):
		qNorm += d_i[i].tfidf ** 2
	qNorm = qNorm ** 0.5
	# Finally, calculate cossim = numerator / denominator
	for i in xrange(doc_nums.size()):
		doc_scores[i].second /= qNorm * X_norm[i]
	# Categorize each doc & its respective scores
	cdef Score scores
	categorize(doc_scores, tY, scores)
	# Return the k-NN (aka top-k similar examples)
	cdef size_t kk
	if k < doc_scores.size():
		kk = k
	else:
		kk = doc_scores.size()
	# Use np.ndarray.partition to partial_sort the first `kk` items
	# the two negation steps are there b/c np.ndarray.partition sorts
	# in ascending order, and we want descending order
	cdef np.ndarray[DSPair] temp = doc_scores
	temp['score'] = -temp['score']
	temp.partition(range(kk), axis=0, order=['score'])
	temp['score'] = -temp['score']
	# obtain pscores
	cdef Score pscores
	get_pscores(parents_index, children_index, scores, pscores)


cdef void categorize(DSPair[:]& doc_scores, object& tY,
	Score& scores):
	'''	Categorize each doc and its respective score '''
	# Categorize each doc and its respective score
	cdef uint[:] labels
	for i in xrange(len(doc_scores)):
		labels = tY.__get_Y(doc_scores[i].first)[:,1]
		for label in labels:
			scores[label].push_back(doc_scores[i].second)


def get_pscores(Family& parents_index, Family& children_index,
	Score& scores, Score& pscores):
	''' Merge child-categories into their parent-categories. For each parent
		category, take the cardinality of its children as its "pscore" '''
	cdef uint label
	cdef Score_it s_it = scores.begin()
	cdef unordered_set[uint] parents
	cdef uint parent
	cdef unordered_set[uint] children
	while s_it != scores.end():
		label = deref(inc2(s_it)).first
		parents = parents_index[label]
		for parent in parents:
			children = children_index[parent]
			pscores[label].push_back(children.size())



cdef void get_candidate_doc_nums(Word[:]& d_i, object& iidx, cset[size_t]& doc_nums):
	''' Find all candidate docs from the words in `d_i` with `iidx` '''
	cdef:
		size_t word
		size_t i
		Family_it got
		unordered_set[size_t]* doc_set
		size_t doc
		cset[size_t] doc_nums
		size_t[:] temp
	for i in xrange(len(d_i)):
		word = d_i[i].word
		temp = iidx.__get_iidx(word)[:,1] # ignore word, get only doc
		for doc in temp:
			doc_nums.insert(doc)


cdef flt csum(vector[flt]& vect):
	''' custom sum func for vector[flt]'s '''
	cdef flt ansatz = 0
	cdef size_t i
	for i in xrange(vect.size()):
		ansatz += vect[i]
	return ansatz


cdef void ranks(Score& scores, Score& pscores, unordered_map[size_t,size_t]& label_counter,
	flt w1, flt w2, flt w3, flt w4, unordered_map[size_t,flt]& ranks):
	''' w1..w4 are weights corresponding to x1..x4 '''
	# What the hell is unordered_map[size_t,size_t]& label_counter?
	cdef:
		pair[uint, vector[flt]] kv
		vector[flt] inner_scores, inner_pscores
		size_t c
		flt x1, x2, x3, x4
		Score_it st = scores.begin()
		while st != scores.end():
			kv = deref(inc2(st))
			c = kv.first
			inner_scores = kv.second
			inner_pscores = pscores[c]
			x1 = deref(max_element_1(inner_scores.begin(), inner_scores.end()))
			x2 = csum(inner_pscores)
			x3 = csum(inner_scores)
			x4 = (<flt>inner_scores.size())/(<flt>label_counter[c])
			ranks[c] = w1*x1 + w2*x2 + w3*x3 + w4*x4


cdef flt custom_max(unordered_map[uint,flt]& lut):
	''' personal func that returns the max value inside an
		unordered_map[uint, flt] '''
	cdef pair[uint,flt] kv
	cdef flt ansatz = deref(lut.begin()).second
	for kv in lut:
		if kv.second > ansatz:
			ansatz = kv.second
	return ansatz


cdef void predict(unordered_map[uint, flt]& ranks, flt alpha, vector[uint]& predictions):
	''' Return a vector of labels if their corresponding ranks are higher than a
		threshold provided by `alpha` '''
	cdef flt max_rank = custom_max(ranks)
	cdef pair[uint,flt]& kv
	unordered_map[uint,flt].iterator rt = ranks.begin()
	while rt != ranks.end():
		kv = deref(inc2(rt))
		if kv.second / max_rank > alpha:
			ans.push_back(kv.first)






	