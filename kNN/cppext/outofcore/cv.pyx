#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
''' 
Cross Validation Module 

Note: CV is not K-fold. Rather, it is "repeated random sub-sampling
validation." Wikipedia states:

	This method randomly splits the dataset into training and validation data. For each such 
	split, the model is fit to the training data, and predictive accuracy is assessed using 
	the validation data. The results are then averaged over the splits. The advantage of this 
	method (over k-fold cross validation) is that the proportion of the training/validation 
	split is not dependent on the number of iterations (folds). The disadvantage of this 
	method is that some observations may never be selected in the validation subsample, 
	whereas others may be selected more than once. In other words, validation subsets may 
	overlap. This method also exhibits Monte Carlo variation, meaning that the results will 
	vary if the analysis is repeated with different random splits.

Source - http://en.wikipedia.org/wiki/Cross-validation_(statistics)

Note: I've cheated a little bit. These are deterministic algorithms. That is, these functions
will always give the same validation and training sub-sets from the given training set.
To gaurantee a new CV-split in every trial, one must manually shuffle X/tfidfX before calling
this function.

A K-fold CV function is also provided.
'''

import numpy as np
from itertools import izip, islice
from collections import defaultdict
import tables as tb

from libcpp.vector cimport vector
from cpython cimport bool
from kNN.cppext.container cimport unordered_map, unordered_set


def even_sample_CV(X_doc_len_idx, h5name='', X=None, Y=None, vX=None, vY=None, tX=None, tY=None, 
		unsigned intmax_per_category=1):
	'''
	As it is observed that the testing data was sampled evenly on
	each category, we perform the same sampling on the training data.
	The resulting validating set consists of one randomly selected
	document from each category in the training data, and the rest of
	the documents are divided into sub-training set.
	'''
	if h5name:
		f = tb.openFile(h5name, mode='r+')
		X, Y = f.root.tfidfX, f.root.Y
		vX, vY = f.root.vX, f.root.vY
		tX, tY = f.root.tX, f.root.tY
	else:
		if (not X) or (not Y) or (not vX) or (not vY) or (not tX) or (not tY):
			raise AssertionError("if h5name is not provided, please provide" \
				"X, Y, vX, vY, tX, and tY")
	cdef unsigned int indleftX, indleftY, indrightX, indrightY
	indleftX, indleftY, indrightX, indrightY = 0, 0, 0, 0
	cdef unordered_map[unsigned int, unsigned int] label_progress
	cdef vector[unsigned int] labels
	cdef unsigned int doc_id, i
	cdef unsigned int curr_doc_id = Y[0]['doc_id']
	cdef bool flag
	# "iterate" by `doc_id` in Y and X
	# then decide whether to append in validation or training
	for r in Y:
		doc_id = r['doc_id']
		label = r['label']
		if curr_doc_id == doc_id:
			if label_progress.find(label) == label_progress.end():
				label_progress[label] = 0
			label_progress[label] += 1
			labels.push_back(label)
			indrightY += 1
		else:
			labels.clear()
			flag = False
			indrightX = indleftX + X_doc_len_idx[doc_id]
			for i in xrange(labels.size()):
				if label_progress[labels[i]] <= max_per_category:
					flag = True
					break
			if flag:
				vY.append(Y[indleftY : indrightY])
				vX.append(X[indleftX : indrightX])
			else:
				tY.append(Y[indleftY : indrightY])
				tX.append(X[indleftX : indrightX])
			indrightY += 1
			indleftY = indrightY
			curr_doc_id = doc_id
		indleftX = indrightX
	# test the last row since it's not included above
	indrightX = indleftX + X_doc_len_idx[doc_id]
		flag = False
		for i in xrange(labels.size()):
			if label_progress[labels[i]] <= max_per_category:
				flag = True
				break
		if flag:
			vY.append(Y[indleftY : indrightY])
			vX.append(X[indleftX : indrightX])
		else:
			tY.append(Y[indleftY : indrightY])
			tX.append(X[indleftX : indrightX])
	vX.flush(); vY.flush(); tX.flush(); tY.flush()
	if h5name:
		f.close()


def prop_sample_CV(X_doc_len_idx, label_counter, h5name='', X=None, Y=None, vX=None, vY=None, tX=None, tY=None, 
		double prop=0.1):
	'''
	Sub-sample according to the proportion of cat populations.
	Use this function if the training population is inbalanced.
	Create a validation and sub-training set from infile.

	Note: This is a deterministic algorithm. That is, `prop_sample_CV` will always give the same validation
	and training sub-sets from the given training set. To gaurantee a new CV-split in every trial, one must
	manually shuffle X/tfidfX before calling this function.
	'''
	if h5name:
		f = tb.openFile(h5name, mode='r+')
		X, Y = f.root.tfidfX, f.root.Y
		vX, vY = f.root.vX, f.root.vY
		tX, tY = f.root.tX, f.root.tY
	else:
		if (not X) or (not Y) or (not vX) or (not vY) or (not tX) or (not tY):
			raise AssertionError("if h5name is not provided, please provide" \
				"X, Y, vX, vY, tX, and tY")
	cdef unsigned int indleftX, indleftY, indrightX, indrightY
	indleftX, indleftY, indrightX, indrightY = 0, 0, 0, 0
	cdef unordered_map[unsigned int, unsigned int] label_progress
	cdef vector[unsigned int] labels
	cdef unsigned int doc_id, label, i
	cdef unsigned int curr_doc_id = Y[0]['doc_id']
	cdef bool flag
	# "iterate" by `doc_id` in Y and X
	# then decide whether to append in validation or training
	for r in Y:
		doc_id = r['doc_id']
		label = r['label']
		if curr_doc_id == doc_id:
			if label_progress.find(label) == label_progress.end():
				label_progress[label] = 0
			label_progress[label] += 1
			labels.push_back(label)
			indrightY += 1
		else:
			labels.clear()
			flag = False
			indrightX = indleftX + X_doc_len_idx[doc_id]
			for i in xrange(labels.size()):
				label = labels[i]
				if label_progress[label] <= prop*label_counter[label]:
					flag = True
					break
			if flag:
				vY.append(Y[indleftY : indrightY])
				vX.append(X[indleftX : indrightX])
			else:
				tY.append(Y[indleftY : indrightY])
				tX.append(X[indleftX : indrightX])
			indrightY += 1
			indleftY = indrightY
			curr_doc_id = doc_id
		indleftX = indrightX
	# test the last row since it's not included above
	indrightX = indleftX + X_doc_len_idx[doc_id]
		flag = False
		for i in xrange(labels.size()):
			label = labels[i]
			if label_progress[label] <= prop*label_counter[label]:
				flag = True
				break
		if flag:
			vY.append(Y[indleftY : indrightY])
			vX.append(X[indleftX : indrightX])
		else:
			tY.append(Y[indleftY : indrightY])
			tX.append(X[indleftX : indrightX])
	vX.flush(); vY.flush(); tX.flush(); tY.flush()
	if h5name:
		f.close()


def kfold_CV(X_doc_len_idx, Y_doc_len_idx, label_counter, h5name='', X=None, Y=None, vX=None,
		vY=None, tX=None, tY=None, K=10, subset_choice=0):
	''' K-Fold CV option. Works just like even_sample_CV and prop_sample_CV.
		K = number of splits (the size of the validation split is 1/K size
				of training set)
		subset_choice = which split to designate as validation set
				(0 <= subset_choice < K) '''
	assert X_doc_len_idx.size() == Y_doc_len_idx.size()
	if h5name:
		f = tb.openFile(h5name, mode='r+')
		X, Y = f.root.tfidfX, f.root.Y
		vX, vY = f.root.vX, f.root.vY
		tX, tY = f.root.tX, f.root.tY
	else:
		if (not X) or (not Y) or (not vX) or (not vY) or (not tX) or (not tY):
			raise AssertionError("if h5name is not provided, please provide" \
				"X, Y, vX, vY, tX, and tY")
	# X_doc_len_idx MUST be a vector
	# otherwise change the following implementation
	cdef unsigned int start, stop
	cdef unsigned int subset_size = X_doc_len_idx.size() // K
	if 0 <= subset_choice < K - 1:
		start = subset_choice * subset_size
		stop = start + subset_size
	elif: subset_choice == K - 1:
		start = subset_choice * subset_size
		stop = X_doc_len_idx.size()
	else:
		raise AssertionError("subset_choice = %d, but 0 <= subset_choice < K" % \
			subset_choice " is not true")

	cdef unsigned int i
	cdef unsigned int indleftX, indleftY, indrightX, indrightY
	indleftX, indleftY, indrightX, indrightY = 0, 0, 0, 0
	# left of start -> training
	for i in xrange(start):
		indrightX += X_doc_len_idx[i]
		indrightY += Y_doc_len_idx[i]
	if indleftX != indrightX:
		tX.append(X[:indrightX])
		tY.append(Y[:indrightY])
	indleftX, indleftY = indrightX, indrightY
	# [start, stop) -> validation
	for i in xrange(start, stop):
		indrightX += X_doc_len_idx[i]
		indrightY += Y_doc_len_idx[i]
	vX.append(X[indleftX : indrightX])
	vY.append(Y[indleftX : indrightY])
	if stop != X_doc_len_idx.size():
		indleftX, indleftY = indrightX, indrightY
		# [stop, end) -> training
		for i in xrange(stop, X_doc_len_idx.size()):
			indrightX += X_doc_len_idx[i]
			indrightY += Y_doc_len_idx[i]
		tX.append(X[indleftX : indrightX])
		tY.append(Y[indleftY : indrightY])

	vX.flush(); vY.flush(); tX.flush(); tY.flush()
	if h5name:
		f.close()

