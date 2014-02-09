from itertools import izip
from collections import defaultdict
import heapq
import operator
import numpy as np

def transform_tfidf(corpus, counter):
	''' Transform corpus into modified tf-idf representation.
		t_corpus = transform_corpus(corpus, counter)

		Use to transform both the test and train set.'''
	for doc in corpus:
		for w in doc:
			doc[w] = np.log(d[w] + 1) * np.log(n / counter[w])

def cossim(t_d, k, t_corpus, counter, labels, parents_index, children_index):
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
		Make sure that `t_corpus` is the already-transformed `corpus`.
		`scores` is a dict key-grouped by category
		`pscores` is a dict key-grouped by category

	'''
	doc_scores, cat_scores_dict = [], defaultdict(list)
	for i,(doc, ls) in enumerate(izip(t_corpus, labels)):
		# Calculate numerator efficiently
		if len(doc) <= len(t_d):
			first, second = doc, t_d
		else:
			first, second = t_d, doc
		top = 0
		for k in first:
			if k in second:
				top += first[k]*second[k]
		# Calculate denominator efficiently
		bottom = (sum([v**2 for v in d_i.itervalues()]) *\
			sum([v**2 for v in doc.itervalues()]))**0.5

		score = top / bottom
		for l in ls:
			doc_scores.append((l, score))
			cat_scores_dict[l].append(score)
	# Return the k-NN (aka top-k similar examples)
	top_k_tups = heapq.nlargest(k, doc_scores, operator.itemgetter(1))
	# optimized/transformed scores & pscores
	scores, pscores = defaultdict(list), defaultdict(list)
	for l, score in top_k_tups:
		scores[l].append(np.log(1 + score))
		pscores[l] = [np.log(len(children_index[parent]))
			for parent in parents_index[l]]

	return scores, pscores





