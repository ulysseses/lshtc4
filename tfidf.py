import numpy as np
from numpy.linalg import norm

def transform_corpus(corpus, counter):
	''' Transform corpus into modified tf-idf representation.
		t_corpus = transform_corpus(corpus, counter)'''
	for doc in corpus:
		for w in doc:
			doc[w] = np.log(d[w] + 1) * np.log(n / counter[w])

def cossim(d, t_corpus, counter):
	'''
	def w(t,d, corpus, counter):

		def tf(t,d):
			return corpus[d][t]

		def idf(t):
			return np.log(n / counter[t])

		return np.log(tf(t,d) + 1) * idf(t)

	def cossim(d_i, d):
		return np.dot(d_i, d) / (np.linalg.norm(d_i) * np.linalgnorm(d))

	This version of cossim, however, takes in `d` as a dictionary, since each
	example is extremely sparse such that the built-in dictionary is faster than
	scipy.sparse matrices

	Make sure that `t_corpus` is the already-transformed `corpus`.
	'''
	# Transform `d` to modified tf-idf
	for w in d:
		d[w] = np.log(d[w] + 1) * np.log(n / counter[w])

	for doc in t_corpus:
		# Calculate numerator efficiently
		if len(doc) <= len(d):
			first, second = doc, d
		else:
			first, second = d, doc
		top = 0
		for k in first:
			if k in second:
				top += first[k]*second[k]
		# Calculate denominator efficiently
		bottom = (sum([v**2 for v in d_i.itervalues()]) *\
			sum([v**2 for v in doc.itervalues()]))**0.5

		




		