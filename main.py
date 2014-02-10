#!/usr/bin/env python
import pickle
from itertools import izip

import preproc
import pruning
import similarity
import cv
import evaluation

if __name__ == '__main__':
	# Test on toyset.
	preproc.subset("raw_data/train.csv", "data/train.csv", 0, 200000)

	# Load toyset .csv -> X & Y
	X, Y = preproc.extract_XY("data/train.csv")

	# Load hierarchy (parents & children indices)
	parents_index = preproc.extract_parents("hierarchy.txt")
	children_index = preproc.inverse_index(parents_index)

	# Prune corpora
	label_counter = pruning.LabelCounter(Y)
	word_counter = pruning.WordCounter(X)
	label_counter.prune(no_below=2, no_above=1.0, max_n=None)
	word_counter.prune(no_below=2, no_above=0.4, max_n=None) # assume balanced
	pruning.prune_corpora(X, Y, label_counter, word_counter)
	del label_counter, word_counter # free up memory

	# Transform X to tf-idf
	bin_word_counter = pruning.WordCounter(X, binary=True)
	similarity.transform_tfidf(X, bin_word_counter)

	# CV-split X & Y (using default params)
	##Save state
	with open("working/X.pickle", 'wb') as picklefile:
		pickle.dump(X, picklefile, -1)
	with open("working/Y.pickle", 'wb') as picklefile:
		pickle.dump(Y, picklefile, -1)
	v_X, v_Y, t_X, t_Y = cv.prop_sample_CV(X=X, Y=Y)
	del X, Y # free up memory

	# Obtain k-NN scores & pscores, predict, and calculate F1!
	k = 70
	w1, w2, w3, w4 = 3.4, 0.6, 0.8, 0.2
	alpha = 0.9
	cat_pns = evaluation.CategoryPNCounter()
	for d_i, labels_i in izip(v_X, v_Y):
		scores, pscores = similarity.cossim(d_i, t_X, k, t_Y, parents_index,
			children_index)
		ranks = similarity.optimized_ranks(scores, pscores, label_counter,
			w1, w2, w3, w4)
		predicted_labels = similarity.predict(ranks, alpha)
		cat_pns.fill_pns(predicted_labels, labels_i)
	cat_pns.calculate_cat_pr()
	MaF = cat_pns.calculate_MaF()

	print "MaF:", MaF











