# hinter.pyx
#distutils: language = c++

import tables as tb
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, \
	postincrement as inc2

cdef extern from "utils.hpp":
	ctypedef unsigned int DOC
	ctypedef unsigned int WORD
	ctypedef unsigned int CAT
	ctypedef unsigned int PCAT
	ctypedef float SCORE
	ctypedef float TFIDF

# def vector_generator(vector[size_t]& y_starts):
# 	cdef size_t i
# 	for i in xrange(y_starts.size()):
# 		yield y_starts[i]

cdef public vector[vector[DOC]] *word2doc_func(char *h5name):
	'''
		word2doc_func returns word2doc
	'''
	f = tb.openFile(h5name, 'r')
	x = f.root.X

	cdef vector[vector[DOC]] *word2doc = new vector[vector[DOC]]()
	cdef WORD word
	for r in x:
		word = r['word']
		if word >= deref(word2doc).size():
			deref(word2doc).resize(word+1, vector[DOC]())
		deref(word2doc)[word].push_back(<DOC>r['doc'])
	return word2doc

# cdef api vector[size_t] *x_starts_func(char *h5name, char x_or_y):
# 	'''
# 		x_starts_func returns x_starts
# 		x_or_y determines what the corpus is
# 	'''
# 	f = tb.openFile(h5name, 'r');
# 	if x_or_y.lower() == 'x':
# 		corpus = f.root.X
# 	elif x_or_y.lower() == 'y':
# 		corpus = f.root.Y

# 	cdef vector[size_t] *x_starts = new vector[size_t]()
# 	cdef DOC doc
# 	cdef DOC curr_doc = corpus[0]['doc']
# 	deref(x_starts).push_back(0);
# 	cdef size_t i

# 	for i, r in enumerate(corpus):
# 		doc = r['doc']
# 		if doc != curr_doc:
# 			curr_doc = doc
# 			deref(x_starts).push_back(i)
# 	f.close()
# 	return x_starts

# cdef api vector[size_t] *x_lens_func(char *h5name, char x_or_y):
# 	'''
# 		x_lens_func returns x_lens
# 		x_or_y determines what the corpus is
# 	'''
# 	f = tb.openFile(h5name, 'r');
# 	if x_or_y.lower() == 'x':
# 		corpus = f.root.X
# 	elif x_or_y.lower() == 'y':
# 		corpus = f.root.Y

# 	cdef vector[size_t] *x_lens = new vector[size_t]()
# 	cdef DOC doc
# 	cdef DOC curr_doc = corpus[0]['doc']
# 	cdef size_t doc_len = 0

# 	for r in corpus:
# 		doc = r['doc']
# 		if doc == curr_doc:
# 			doc_len += 1
# 		else:
# 			if doc >= x_lens.size():
# 				deref(x_lens).resize(doc+1)
# 			deref(x_lens)[doc] = doc_len
# 			doc_len = 0
# 			curr_doc = doc
# 	f.close()
# 	return x_lens

# cdef api vector[vector[CAT]] *doc2cat_func(char* h5name, vector[size_t]& y_starts,
# 		vector[size_t]& y_lens):
# 	'''
# 		doc2cat_func returns doc2cat.
# 		y is the h5table handle.
# 		y_starts & y_lens accesses y.
# 	'''
# 	f = tb.openFile(h5name, 'r')
# 	y = f.root.Y
# 	cdef vector[vector[CAT]] *doc2cat = new vector[vector[CAT]]()
# 	cdef vector[CAT] *inner
# 	cdef size_t j, k
# 	for j, row in enumerate(y.iterrows(vector_generator(y_starts))):
# 		deref(doc2cat).push_back(vector[CAT]())
# 		inner = &(deref(doc2cat).back())
# 		for k in xrange(y_lens[j]):
# 			inner.push_back(<CAT>row['label'])
# 	f.close()
# 	return doc2cat
