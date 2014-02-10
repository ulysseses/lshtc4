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

A K-fold CV function is also provided, if desired...
'''

import numpy as np
from itertools import izip, islice


def even_sample_CV(iname=None, otemp=None, out_of_core=False, X=None, Y=None):
	'''
	As it is observed that the testing data was sampled evenly on
	each category, we perform the same sampling on the training data.
	The resulting validating set consists of one randomly selected
	document from each category in the training data, and the rest of
	the documents are divided into sub-training set.
	'''
	if out_of_core:
		# Create mapping of label -> file line number
		v_label_dict = create_label_num_map(iname, True)
		# Sample 1 time each out of cat population
		v_line_set = set()
		for label, ns in v_label_dict.iteritems():
			cat_pop = len(ns)
			v_line_set.add(inplace_choice(ns, 1).next())
		# Store examples whose line number is in v_line_set into validation
		# Store the rest into sub-training
		separate_vt(iname, otemp, True, v_line_set)
	else:
		if not X:
			X,Y = extract_XY(iname)
		# Create mapping of label -> file line number
		v_label_dict = create_label_num_map(Y=Y)
		# Sample `prop` times each cat population
		v_line_set = set()
		for label, ns in v_label_dict.iteritems():
			cat_pop = len(ns)
			v_line_set.update([num for num in inplace_choice(ns, int(prop*cat_pop))])
		# Store a list containing ln's in v_line_set into validation
		# Store the rest into sub-training
		return separate_vt(v_line_set=v_line_set, X=X, Y=Y)


def prop_sample_CV(iname=None, otemp=None, prop=0.1, out_of_core=False, X=None, Y=None):
	'''
	Sub-sample according to the proportion of cat populations.
	Use this function if the training population is inbalanced.
	Create a validation and sub-training set from infile.
	'''
	if out_of_core:
		# Create mapping of label -> file line number
		v_label_dict = create_label_num_map(iname, True)
		# Sample 1 time each out of cat population
		v_line_set = set()
		for label, ns in v_label_dict.iteritems():
			cat_pop = len(ns)
			v_line_set.update([num for num in inplace_choice(ns, int(prop*cat_pop))])
		# Store examples whose line number is in v_line_set into validation
		# Store the rest into sub-training
		separate_vt(iname, otemp, True, v_line_set)
	else:
		if not X:
			X,Y = extract_XY(iname)
		# Create mapping of label -> file line number
		v_label_dict = create_label_num_map(Y=Y)
		# Sample `prop` times each cat population
		v_line_set = set()
		for label, ns in v_label_dict.iteritems():
			cat_pop = len(ns)
			v_line_set.update([num for num in inplace_choice(ns, int(prop*cat_pop))])
		# Store a list containing ln's in v_line_set into validation
		# Store the rest into sub-training
		return separate_vt(v_line_set=v_line_set, X=X, Y=Y)


# Get random sample from list while maintaining ordering of items?
def inplace_choice(seq, k):
	pick_count = 0
	for i, val in enumerate(seq):
		prob = (k - pick_count) / (len(seq) - i)
		if np.random.random() < prob:
			yield val
			pick_count += 1


def create_label_num_map(iname=None, out_of_core=False, Y=None):
	''' Helper function to create mapping of label -> file line number '''
	v_label_dict = defaultdict(list)
	if out_of_core:
		with open(iname, 'rb') as i:
			for n, line in enumerate(i):
				line_comma_split = line.split(',')
				labels = line_comma_split[:-1]
				pre_kvs = line_comma_split[-1].split()
				labels.append(pre_kvs[0])
				labels = [int(label) for label in labels]
				for label in labels:
					v_label_dict[label].append(n)
	else:
		for n,labels in enumerate(Y):
			for label in labels:
				v_label_dict[label].append(n)
	return v_label_dict


def separate_vt(iname=None, otemp=None, out_of_core=False, v_line_set=None, X=None, Y=None):
	''' helper function for even_sample_CV and prop_sample_CV. '''
	if out_of_core:
		i = open(iname, 'rb')
		ot, ov = open(otemp % 't', 'wb'), open(otemp % 'v', 'wb')
		with i, ot, ov:
			for n,line in enumerate(i):
				if n in v_line_set:
					ov.write(line + '\n')
				else:
					ot.write(line + '\n')
		return
	else:
		v_X, v_Y, t_X, t_Y = [], [], [], []
		for n,(doc,labels) in enumerate(izip(X, Y)):
			if n in v_line_set:
				v_X.append(doc); v_Y.append(labels)
			else:
				t_X.append(doc); t_Y.append(labels)
		return (v_X, v_Y, t_X, t_Y)

def kfold_CV(iname=None, otemp=None, out_of_core=False, X=None, Y=None,
		K=10, subset_choice=0):
	''' K-Fold CV option. Works just like even_sample_CV and prop_sample_CV. '''
	if out_of_core:
		with open(iname, 'rb') as i:
			for n, line in enumerate(i):
				pass
		line_count = n
		subset_size = line_count // K
		if subset_choice == K - 1:
			subset_size = line_count - (line_count // K)
		start_line_num = subset_choice * line_count // K
		i = open(iname, 'rb')
		ov, ot = open(otemp % 'v', 'wb'), open(otemp % 't', 'wb')
		with i, ov, ot:
			# Skip header
			i.readline();
			# Skip to start_line_num
			for n in xrange(start_line_num):
				ot.write(i.readline())
			for n in xrange(start_line_num, start_line_num+subset_size):
				ov.write(i.readline())
			# Write the rest to sub-training
			ot.write(i.read())
		return
	else:
		line_count = len(Y)
		subset_size = line_count // K
		if subset_choice == K - 1:
			subset_size = line_count - (line_count // K)
		start_line_num = line_count - (line_count // K)
		v_X, v_Y, t_X, t_Y = [], [], [], []
		v_X.extend(X[start_line_num : start_line_num+subset_size])
		v_Y.extend(Y[start_line_num : start_line_num+subset_size])
		try:
			t_X.extend(X[:start_line_num])
			t_Y.extend(Y[:start_line_num])
		except:
			pass
		try:
			t_X.extend(X[start_line_num+subset_size:])
			t_Y.extend(Y[start_line_num+subset_size:])
		except:
			pass
		return (v_X, v_Y, t_X, t_Y)



















