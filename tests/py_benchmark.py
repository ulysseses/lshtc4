#!/usr/bin/env python
import numpy as np
from itertools import izip

import sys
sys.path.insert(0, '..')
from pylib import preproc, pruning, cv, evaluation, similarity


def stage0():
	# Test on toyset.
	preproc.subset("../raw_data/train.csv", "../data/train.csv", 1, 200000)

def stage1():
	# Load toyset .csv -> X & Y
	X, Y = preproc.extract_XY("../data/train.csv")
	# Prune corpora
	label_counter = pruning.LabelCounter(Y)
	word_counter = pruning.WordCounter(X)
	label_counter.prune(no_below=2, no_above=1.0, max_n=None)
	word_counter.prune(no_below=2, no_above=0.4, max_n=None) # assume balanced
	pruning.prune_corpora(X, Y, label_counter, word_counter)
	del word_counter # free up memory
	##Save state
	with open("working/X.dat", 'wb') as picklefile:
		cPickle.dump(X, picklefile, -1)
	with open("working/Y.dat", 'wb') as picklefile:
		cPickle.dump(Y, picklefile, -1)

def stage2():
	with open("working/tX.dat", 'rb') as picklefile:
		X = cPickle.load(picklefile)

	# Transform X to tf-idf
	bin_word_counter = pruning.WordCounter(X, binary=True)
	similarity.transform_tfidf(X, bin_word_counter)
	del bin_word_counter
	##Save state
	with open("working/tX.dat", 'wb') as picklefile:
		cPickle.dump(X, picklefile, -1)

def stage3():
	with open("working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)

	# Load hierarchy (parents & children indices)
	parents_index = preproc.extract_parents(Y, "../raw_data/hierarchy.txt")
	children_index = preproc.inverse_index(parents_index)
	##Save state
	with open("working/parents_index.dat", 'wb') as picklefile:
		cPickle.dump(parents_index, picklefile, -1)
	with open("working/children_index.dat", 'wb') as picklefile:
		cPickle.dump(children_index, picklefile, -1)

def stage4():
	with open("working/tX.dat", 'rb') as picklefile:
		X = cPickle.load(picklefile)
	with open("working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)

	# CV-split X & Y (using default params)
	v_X, v_Y, t_X, t_Y = cv.prop_sample_CV(X=X, Y=Y)
	del X, Y
	##Save state
	with open("working/v_X.dat", 'wb') as picklefile:
		cPickle.dump(v_X, picklefile, -1)
	with open("working/v_Y.dat", 'wb') as picklefile:
		cPickle.dump(v_Y, picklefile, -1)
	with open("working/t_X.dat", 'wb') as picklefile:
		cPickle.dump(t_X, picklefile, -1)
	with open("working/t_Y.dat", 'wb') as picklefile:
		cPickle.dump(t_Y, picklefile, -1)

def loaded_main():
	##rebuild label_counter manually from Y b/c of weird cPickle bug
	##in serializing LabelCounter's
	with open("working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)
	label_counter = pruning.LabelCounter(Y)
	del Y
	with open("working/parents_index.dat", 'rb') as picklefile:
		parents_index = cPickle.load(picklefile)
	with open("working/children_index.dat", 'rb') as picklefile:
		children_index = cPickle.load(picklefile)
	with open("working/v_X.dat", 'rb') as picklefile:
		v_X = cPickle.load(picklefile)
	with open("working/v_Y.dat", 'rb') as picklefile:
		v_Y = cPickle.load(picklefile)
	with open("working/t_X.dat", 'rb') as picklefile:
		t_X = cPickle.load(picklefile)
	with open("working/t_Y.dat", 'rb') as picklefile:
		t_Y = cPickle.load(picklefile)

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


def onego_main():
	# Test on toyset.
	preproc.subset("../raw_data/train.csv", "../data/train.csv", 1, 200000)

	# Load toyset .csv -> X & Y
	X, Y = preproc.extract_XY("../data/train.csv")

	# Prune corpora
	label_counter = pruning.LabelCounter(Y)
	word_counter = pruning.WordCounter(X)
	label_counter.prune(no_below=2, no_above=1.0, max_n=None)
	word_counter.prune(no_below=2, no_above=0.4, max_n=None) # assume balanced
	pruning.prune_corpora(X, Y, label_counter, word_counter)
	del word_counter # free up memory

	# Transform X to tf-idf
	bin_word_counter = pruning.WordCounter(X, binary=True)
	similarity.transform_tfidf(X, bin_word_counter)
	del bin_word_counter # free up memory

	# Load hierarchy (parents & children indices)
	parents_index = preproc.extract_parents(Y, "../raw_data/hierarchy.txt")
	children_index = preproc.inverse_index(parents_index)

	# CV-split X & Y (using default params)
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

if __name__ == '__main__':
	if len(argv) == 1:
		onego_main()
	else:
		stage = int(argv[1])
		if stage == 0: stage0()
		if stage == 1: stage1()
		if stage == 2: stage2()
		if stage == 3: stage3()
		if stage == 4: stage4()
		if stage == 5: loaded_main()

