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
from __future__ import division
from container cimport unordered_map, unordered_set



def even_sample_CV(iname, oname_t, oname_v):
	'''
	As it is observed that the testing data was sampled evenly on
	each category, we perform the same sampling on the training data.
	The resulting validation set consists of one randomly selected
	document from each category in the training data, and the rest of
	the documents are divided into sub-training set.

	`iname`   - libsvm filename
	`oname_t` - training libsvm output filename
	`oname_v` - validation libsvm output filename

	'''
	cdef unordered_set[LABEL] seen_labels
	infile, outfile_t, outfile_v = open(iname, 'rb'), open(oname_t, 'wb'), open(oname_v, 'wb')
	with infile, outfile_t, outfile_v:
		for line in infile:								# "545, 32 8:1 18:2"
			# Get labels
			line_comma_split = line.split(',') 			# ['545', ' 32 8:1 18:2']
			labels = line_comma_split[:-1] 				# ['545']
			pre_kvs = line_comma_split[-1].split() 		# ['32', '8:1', '18:2']
			labels.append(line_comma_split[-1].split()[0]) # ['545', '32']

			# Test membership: if a label in a doc's labels hasn't been
			# seen yet then put it into the validation set
			unique = False
			for label in labels:
				if seen_labels.find(int(label)) == seen_labels.end():
					seen_labels.insert(int(label))
					unique = True
			if unique:
				outfile_v.write(line + '\n')
			else:
				outfile_t.write(line + '\n')

def prop_sample_CV(iname, oname_t, oname_v, label_counter, prop=0.1):
	'''
	Sub-sample according to the proportion of the category population.
	Use this function if the training population is imbalanced.
	Create a validation and sub-training set from infile.

	Note: This is a deterministic algorithm. That is, `prop_sample_CV` will always give
	the same validation and training sub-sets from the given training set. To gaurantee
	a new CV-split in every trial, one must manually shuffle the dataset before calling
	this function.
	'''
	cdef unordered_map[LABEL, size_t] label_progress

	infile, outfile_t, outfile_v = open(iname, 'rb'), open(oname_t, 'wb'), open(oname_v, 'wb')
	with infile, outfile_t, outfile_v:
		for line in infile:								# "545, 32 8:1 18:2"
			# Get labels
			line_comma_split = line.split(',') 			# ['545', ' 32 8:1 18:2']
			labels = line_comma_split[:-1] 				# ['545']
			pre_kvs = line_comma_split[-1].split() 		# ['32', '8:1', '18:2']
			labels.append(line_comma_split[-1].split()[0]) # ['545', '32']

			# Account for each label. If the label current population is equal or greater
			# than prop * label actual population, then don't put into validation.
			for label in labels:
				#if label_progress.find(int(label)) != label_progress.end():
				label_progress[int(label)] += 1
				if label_progress[int(label)] < prop * label_counter[int(label)]:
					outfile_v.write(line + '\n')
				else:
					outfile_t.write(line + '\n')

def subset(iname, oname, offset, lines):
	''' split a file into 2, starting at `offset`, of size `lines`,
		from file `iname` to file `oname`. '''
	i, o = open(iname, 'rb'), open(oname, 'wb')
	with i,o:
		for l in islice(i, offset, offset+lines):
			o.write(l + '\n')

def kfold_CV(iname, oname_t, oname_v, N, K=10, subset_choice=0):
	''' K-Fold CV option. Works just like even_sample_CV and prop_sample_CV.
		N = number of total training documents
		K = number of splits (the size of the validation split is 1/K size
				of training set)
		subset_choice = which split to designate as the validation set
				(0 <= subset_choice < K) '''
	# determine where the start and ending of the validation set should be
	# from within the total training set
	subset_size = N // K
	if 0 <= subset_choice < K - 1:
		start = subset_choice * subset_size
		stop = start + subset_size
	elif subset_choice == K - 1:
		start = subset_choice * subset_size
		stop = N
	else:
		raise AssertionError("subset_choice = %d, but 0 <= subset_choice < K" % \
			subset_choice " is not true.")
	if start == 0:
		subset(iname, oname_v, start, stop-start)
		subset(iname, oname_t, stop, N)
	elif stop == N:
		subset(iname, oname_v, start, stop-start)
		subset(iname, oname_t, 0, start)
	else:
		i, o_t, o_v = open(iname, 'rb'), open(oname_t, 'wb'), open(oname_v, 'wb')
		with i, o_t, o_v:
			for n,l in enumerate(i):
				if start <= n < stop:
					o_v.write(l + '\n')
				else:
					o_t.write(l + '\n')

def shuffle_file(iname, oname):
	''' In-memory solution to randomly shuffle one file into another.
		Please refer to http://stackoverflow.com/questions/4618298/randomly-mix-lines-of-3-million-line-file '''
	i,o = open(iname, 'rb'), open(oname, 'wb')
	with i,o:
		import random
		lines = i.readlines()
		random.shuffle(lines)
		o.writelines(lines)

def sort_libsvm_hashes(iname, oname):
	''' The raw libsvm file is (presumably) unsorted in both the labels and the hashes.
		`sort_libsvm_hashes` will sort in ascending order both of them for each line. '''
	i,o = open(iname, 'rb'), open(oname, 'wb')
	with i,o:
		for line in enumerate(i):
			# 1st, extract out the kvs and labels into pythonic structures
			line_comma_split = line.split(',')
			labels = line_comma_split[:1]
			pre_kvs = line_comma_split[-1].split()
			labels.append(pre_kvs[0])
			labels = [int(label) for label in labels]
			pre_kvs = pre_kvs[1:]
			kvs = {}
			for kv_str in pre_kvs:
				k,v = kv_str.split(':')
				kvs[int(k)] = int(v)
			# 2nd, sort both kvs and labels
			labels.sort()
			sorted_kv_items = [tup for tup in sorted(kvs.iteritems())]
			sorted_kv_strs = ["%d:%d" % (k,v) for (k,v) in sorted_kv_items]
			# 3rd, write back in oname
			o.write(", ".join(labels) + " " + " ".join(sorted_kv_strs) + '\n')

