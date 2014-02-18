from __future__ import division
from itertools import izip
from collections import defaultdict
import heapq
import operator
import numpy as np
from numpy.linalg import norm

def transform_tfidf(corpus, bin_word_counter):
	''' Transform corpus into modified tf-idf representation.
		t_X = transform_corpus(corpus, bin_word_counter)

		Use to transform both the test and train set.'''
	n = len(corpus)
	for doc in corpus:
		for w in doc:
			doc[w] = np.float32(np.log(doc[w] + 1) *\
				np.log(n / bin_word_counter[w]))

def cossim(d_i, t_X, k, t_Y, parents_index, children_index):
	'''
	def w(t,d, corpus, counter):

		def tf(t,d):
			return corpus[d][t]

		def idf(t):
			return np.log(n / counter[t])

		return np.log(tf(t,d) + 1) * idf(t)

	def cossim(d_i, d):
		return np.dot(d_i, d) / (np.linalg.norm(d_i) * np.linalgnorm(d))

	------------------------------------------------------------------------
	This version of cossim, however, takes in `d` as a dictionary, since each
	example is extremely sparse such that the built-in dictionary is faster than
	scipy.sparse matrices. Also reduces dimension to k-NN.

	Returns a pair: `scores`, `pscores`.

	NOTE:

		c <-> l <-> label <-> child <-> cat <-> category
		Make sure that `t_X` is the already-transformed `corpus`.
		`scores` is a dict key-grouped by category
		`pscores` is a dict key-grouped by category

	'''
	doc_scores, cat_scores_dict = [], defaultdict(list)
	for doc, labels in izip(t_X, t_Y):
		# Calculate numerator efficiently
		if len(doc) <= len(d_i):
			first, second = doc, d_i
		else:
			first, second = d_i, doc
		top = 0
		for word in first:
			if word in second:
				top += first[word]*second[word]
		# Calculate denominator efficiently
		bottom = norm(d_i.values())*norm(doc.values())

		score = top / bottom
		for label in labels:
			doc_scores.append((label, score))
			cat_scores_dict[label].append(score)
	# Return the k-NN (aka top-k similar examples)
	top_k_tups = heapq.nlargest(k, doc_scores, operator.itemgetter(1))
	# optimized/transformed scores & pscores
	scores, pscores = defaultdict(list), defaultdict(list)
	for label, score in top_k_tups:
		#scores[label].append(np.log(1 + score))
		scores[label].append(score)
		# pscores[label] = [np.log(len(children_index[parent]))
		# 	for parent in parents_index[label]]
		pscores[label] = [len(children_index[parent])
			for parent in parents_index[label]]

	return scores, pscores

def optimized_ranks(scores, pscores, label_counter, w1, w2, w3, w4):
	''' w1..w4 are weights corresponding to x1..x4 '''
	ranks_dict = {}
	for c in scores:
		# x1 = np.log(max(scores[c]))
		# x2 = np.log(sum(pscores[c]))
		# x3 = np.log(sum(scores[c]))
		# x4 = np.log(len(scores[c])/label_counter[c])
		x1 = max(scores[c])
		x2 = sum(pscores[c])
		x3 = sum(scores[c])
		x4 = len(scores[c])/label_counter[c]
		ranks_dict[c] = w1*x1 + w2*x2 + w3*x3 + w4*x4
	return ranks_dict

def predict(ranks, alpha):
	''' Return a list of labels if their corresponding ranks are higher
		than a threshold provided by `alpha`.
	'''
	max_rank = max(ranks.itervalues())
	return [label for label, rank in ranks.iteritems() if rank/max_rank > alpha]




