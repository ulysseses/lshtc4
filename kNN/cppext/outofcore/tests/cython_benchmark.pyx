#distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False
'''
Using Python interpretor within root directory:
	
	from kNN.cppext.tests import cython_benchmark as cb
	cb.stage0(raw="...", out="...")
	cb.stage1(...)

'''

import cPickle
from kNN.pyext.incore import preproc, pruning, cv
from kNN.pyext.incore import similarity as py_similarity
from kNN.pyext import benchmark
from kNN.pyext import evaluation as py_evaluation

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc
from kNN.cppext.container cimport unordered_map, unordered_set
from kNN.cppext import evaluation
from kNN.cppext.incore cimport similarity, convert

@benchmark.print_time
def stage0(raw="../raw_data/train.csv", out="../data/train.csv", start=1, stop=200000):
	# Test on toyset
	preproc.subset(raw, out, start, stop)

@benchmark.print_time
def stage1(fh="../data/train.csv", nbl=3, nbw=3, nal=1.0, naw=0.4, mnl=None, mnw=None):
	# Load toyset .csv -> X & Y
	X, Y = preproc.extract_XY(fh)
	print "extract_XY done"
	# Prune corpora
	label_counter = pruning.LabelCounter(Y)
	word_counter = pruning.WordCounter(X)
	label_counter.prune(no_below=nbl, no_above=nal, max_n=mnl)
	word_counter.prune(no_below=nbw, max_n=mnw)
	pruning.prune_corpora(X, Y, label_counter, word_counter)
	print "prune_corpora done"
	##Save state
	with open("../working/word_counter.dat", 'wb') as picklefile:
		cPickle.dump(word_counter, picklefile, -1)
	del word_counter
	print "pickling word_counter done"
	with open("../working/X.dat", 'wb') as picklefile:
		cPickle.dump(X, picklefile, -1)
	del X
	print "pickling X done"
	with open("../working/Y.dat", 'wb') as picklefile:
		cPickle.dump(Y, picklefile, -1)
	del Y
	print "pickling Y done"

@benchmark.print_time
def stage2(nbl=3, nbw=3, nal=1.0, naw=0.4, mnl=None, mnw=None):
	# Prune 2nd stage
	with open("../working/X.dat", 'rb') as picklefile:
		X = cPickle.load(picklefile)
	with open("../working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)
	label_counter = pruning.LabelCounter(Y)
	bin_word_counter = pruning.WordCounter(X, binary=True)
	bin_word_counter.prune(no_above=naw, max_n=mnw)
	pruning.prune_corpora(X, Y, label_counter, bin_word_counter)
	del label_counter
	with open("../working/Y.dat", 'wb') as picklefile:
		cPickle.dump(Y, picklefile, -1)
	del Y
	with open("../working/bin_word_counter.dat", 'wb') as picklefile:
		cPickle.dump(bin_word_counter, picklefile, -1)
	del bin_word_counter
	iidx = preproc.inverse_index(X)
	with open("../working/iidx.dat", 'wb') as picklefile:
		cPickle.dump(iidx, picklefile, -1)
	del iidx
	with open("../working/X.dat", 'wb') as picklefile:
		cPickle.dump(X, picklefile, -1)


@benchmark.print_time
def stage3():
	with open("../working/X.dat", 'rb') as picklefile:
		X = cPickle.load(picklefile)

	# Transform X to tf-idf
	bin_word_counter = pruning.WordCounter(X, binary=True)
	py_similarity.transform_tfidf(X, bin_word_counter)
	del bin_word_counter
	##Save state
	with open("../working/tX.dat", 'wb') as picklefile:
		cPickle.dump(X, picklefile, -1)

@benchmark.print_time
def stage4(hierarchy_handle="../raw_data/hierarchy.txt"):
	with open("../working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)

	# Load hierarchy (parents & children indices)
	parents_index = preproc.extract_parents(Y, hierarchy_handle)
	children_index = preproc.invert_index(parents_index)
	##Save state
	with open("../working/parents_index.dat", 'wb') as picklefile:
		cPickle.dump(parents_index, picklefile, -1)
	with open("../working/children_index.dat", 'wb') as picklefile:
		cPickle.dump(children_index, picklefile, -1)

@benchmark.print_time
def stage5():
	''' At the moment, for this stage, you'll have to directly modify which 
	cv function you'll want to use to split X/Y into their respective 
	validation/training sub-sets. In the future, the user will be able to 
	easily select which cv-strategy to invoke by passing args/kwargs. '''
	with open("../working/tX.dat", 'rb') as picklefile:
		X = cPickle.load(picklefile)
	with open("../working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)

	# CV-split X & Y (using default params)
	v_X, v_Y, t_X, t_Y = cv.prop_sample_CV(X=X, Y=Y)
	del X, Y
	##Save state
	with open("../working/v_X.dat", 'wb') as picklefile:
		cPickle.dump(v_X, picklefile, -1)
	with open("../working/v_Y.dat", 'wb') as picklefile:
		cPickle.dump(v_Y, picklefile, -1)
	with open("../working/t_X.dat", 'wb') as picklefile:
		cPickle.dump(t_X, picklefile, -1)
	with open("../working/t_Y.dat", 'wb') as picklefile:
		cPickle.dump(t_Y, picklefile, -1)

@benchmark.print_time
def loaded_main(int n_iterations=-1, int k=70, double w1=3.4, double w2=0.6,
		double w3=0.8, double w4=0.2, double alpha=0.76):
	##rebuild label_counter manually to avoid weird cPickle bug
	with open("../working/Y.dat", 'rb') as picklefile:
		Y = cPickle.load(picklefile)
	label_counter = pruning.LabelCounter(Y)
	top2labels = [label for label,count in label_counter.most_common(2)]
	del Y
	with open("../working/iidx.dat", 'rb') as picklefile:
		iidx = cPickle.load(picklefile)
	with open("../working/parents_index.dat", 'rb') as picklefile:
		parents_index = cPickle.load(picklefile)
	with open("../working/children_index.dat", 'rb') as picklefile:
		children_index = cPickle.load(picklefile)
	with open("../working/v_X.dat", 'rb') as picklefile:
		vX = cPickle.load(picklefile)
	with open("../working/v_Y.dat", 'rb') as picklefile:
		vY = cPickle.load(picklefile)
	with open("../working/t_X.dat", 'rb') as picklefile:
		tX = cPickle.load(picklefile)
	with open("../working/t_Y.dat", 'rb') as picklefile:
		tY = cPickle.load(picklefile)
	# Convert Pythonic containers to Cythonic containers
	cdef vector[unordered_map[int,double]] c_vX = convert.cythonize_X(vX)
	del vX
	# cdef vector[vector[int]] c_vY = convert.cythonize_Y(vY)
	# del vY
	cdef vector[unordered_map[int,double]] c_tX = convert.cythonize_X(tX)
	del tX
	cdef vector[vector[int]] c_tY = convert.cythonize_Y(tY)
	del tY
	cdef unordered_map[int, unordered_set[int]] c_iidx = \
		convert.cythonize_iidx(iidx)
	del iidx
	cdef unordered_map[int, unordered_set[int]] c_parents_index = \
		convert.cythonize_index(parents_index)
	del parents_index
	cdef unordered_map[int, unordered_set[int]] c_children_index = \
		convert.cythonize_index(children_index)
	del children_index
	cdef unordered_map[int,int] c_label_counter = \
		convert.cythonize_counter(label_counter)
	del label_counter
	if n_iterations == -1:
		n_iterations = c_vX.size()
		print "n_iterations:", n_iterations
	# Dump containers to see if they're constructed correctly
	print "---------------------------------------------------------------------"
	print "value of first word in first doc in c_vX:", deref(deref(c_vX.begin()).begin()).second
	print "first label in the first example (vY):", vY[0][0]
	# print "first label in the first example (c_vY):", deref(deref(c_vY.begin()).begin())
	print "first parent of the first child (c_parents_index):", deref(deref(c_parents_index.begin()).second.begin())
	print "first child of the first parent (c_children_index):", deref(deref(c_children_index.begin()).second.begin())
	print  "count of the first label in c_label_counter:", deref(c_label_counter.begin()).second
	print "num of the first doc that contains the first word in c_iidx:", deref(deref(c_iidx.begin()).second.begin())
	print "---------------------------------------------------------------------"

	@benchmark.print_time
	def stage6():
		# Obtain k-NN scores & pscores, predict, and calculate F1!
		# cdef int n_iterations = 20
		# cdef int k = 70
		# cdef double w1, w2, w3, w4
		# w1, w2, w3, w4 = 3.4, 0.6, 0.8, 0.2
		# cdef double alpha = 0.9
		cat_pns = evaluation.CategoryPNCounter()
		cdef unordered_map[int, double] d_i
		cdef vector[int] labels_i
		cdef vector[unordered_map[int,double]].iterator it = c_vX.begin()
		# cdef vector[vector[int]].iterator it2 = c_vY.begin()
		cdef int i
		cdef pair[unordered_map[int, vector[double]], \
			unordered_map[int, vector[double]]] scores_pair
		cdef unordered_map[int, vector[double]] scores, pscores
		cdef unordered_map[int, double] ranks
		cdef vector[int] predicted_labels
		for i in xrange(n_iterations):
			d_i = deref(it)
			# labels_i = deref(it2)
			inc(it)
			# inc(it2)
			# scores_pair = similarity.cossim(d_i, c_tX, k, c_tY, c_parents_index,
			# 	c_children_index)
			scores_pair = similarity.cossim2(d_i, c_tX, k, c_tY, c_parents_index,
				c_children_index, c_iidx)
			scores = scores_pair.first
			pscores = scores_pair.second
			ranks = similarity.optimized_ranks(scores, pscores, c_label_counter,
				w1, w2, w3, w4)
			if ranks.size() != 0:
				predicted_labels = similarity.predict(ranks, alpha)
				py_pred_labels = [predicted_labels[<int>x]
					for x in xrange(predicted_labels.size())]
			else:
				py_pred_labels = top2labels
			py_labels_i = vY[i]
		# 	py_labels_i = [labels_i[<int>x]
		# 		for x in xrange(labels_i.size())]
			cat_pns.fill_pns(py_pred_labels, py_labels_i)
		cat_pns.calculate_cat_pr()
		MaF = cat_pns.calculate_MaF()
		print "MaF:", MaF

	stage6()