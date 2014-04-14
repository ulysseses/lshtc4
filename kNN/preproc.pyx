#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = True
from __future__ import division

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, \
	postincrement as inc2

from container cimport unordered_map, unordered_set
from types cimport *

import numpy as np
cimport numpy as np

import cPickle as pickle
import blz


cdef void extract_parents(char* infilename, Y, Family& parents_index):
	'''
	Extract the immediate parents_index of each leaf node.
	Builds an index of child->parents_index
	'''
	# Make sure only immediate parents of the bottom-most children (leaves) are considered
	cdef unordered_set[uint] seen_children
	for i in xrange(Y.starts.size()):
		seen_children.insert(Y.ooc_store[Y.starts[i]][1]) # <-- hacky way of accessing `label` within the ooc_store
	cdef uint parent, child
	with open(infilename, 'rb') as f:
		for line in f:
			# guaranteed each line has 2 tokens
			parent, child = [int(x) for x in line.split()]
			if seen_children.find(child) != seen_children.end(): # if child is a seen child, then it is a leaf!
				parents_index[child].insert(parent)


cdef void parents2children(Family& parents_index, Family& children_index):
	'''
	Build an inverse index of parent->children.
	Focus our attention only onto immediate parents_index of leaf nodes.
	No grandparents_index (of leaf nodes) allowed.
	'''
	cdef:
		Family_it it = parents_index.begin()
		pair[uint, unordered_set[uint]] kv
		unordered_set[uint] p_set
		uint p
	while it != parents_index.end():
		kv = deref(inc2(it))
		p_set = kv.second
		for p in p_set:
			children_index[p].insert(kv.first)


cdef void extract_XY(infilename, Xname, Yname, expectedlen=2000000,
	Baggage& X, Baggage& Y):
	'''
	Given a libsvm multi-label formatted training text file, extract the labels
	and (hash, tf) pairs. Store the pairs into a pythonic dict, and store the
	labels into a pythonic list. Then, convert these structures into their
	respective tables.

	Finally, wrap the `blz.btable`s into the `Baggage` class and output
	X and Y as a tuple.
	'''
	Xtable = blz.btable(np.empty(0, dtype='u4,u4,f4'), expectedlen=expectedlen, rootdir=Xname)
	Ytable = blz.btable(np.empty(0, dtype='u4,u4'), expectedlen=expectedlen, rootdir=Yname)
	with open(infilename, 'r') as infile:
		for i,line in enumerate(infile):				# "545, 32 8:1 18:2"
			# 1st, extract out the kvs and labels into pythonic structures
			line_comma_split = line.split(',') 			# ['545', ' 32 8:1 18:2']
			labels = line_comma_split[:-1] 				# ['545']
			pre_kvs = line_comma_split[-1].split() 		# ['32', '8:1', '18:2']
			labels.append(pre_kvs[0]) 					# ['545', '32']
			labels = [int(label) for label in labels] 	# [545, 32]
			pre_kvs = pre_kvs[1:] 						# ['8:1', '18:2']
			kvs = {}
			for kv_str in pre_kvs:
				k,v = kv_str.split(':')
				kvs[int(k)] = int(v) 					# {8:1, 18:2}
			# 2nd, convert the pythonic structures into sparse tabular form
			for k,v in kvs.iteritems():
				Xtable.append((i, k, v))
			for label in labels:
				Ytable.append((i, label))
		Xtable.flush()
		Ytable.flush()

		# Finally, wrap into Baggage
		X = Baggage(Xtable)
		Y = Baggage(Ytable)

cdef void create_iidx(infilename, iidx_name, expectedlen=-1,
	Baggage& iidx):
	'''
	Given a libsvm-format multi-label formatted training text file, create a 
	word->doc mapping, better known as an inverted index (iidx).
	If `expectedlen` is its default value of -1, then this function will
	perform one full pass over the file to calculate the expected length,
	then perform one more pass to actually fill the blz btable.

	Note: This method is potentially memory-intensive.

	'''
	cdef unordered_set[uint] word_set
	with open(infilename, 'rb') as infile:
		# if expectedlen is its default value, calculate expectedlen
		if expectedlen == -1:
			for line in infile:
				line_comma_split = line.split(',')
				pre_kvs = line_comma_split[-1].split()[1:]
				for kv_str in pre_kvs:
					k,v = kv_str.split(':')
					word_set.insert(<uint>int(k))
			expectedlen = word_set.size()
	# Perform a full pass over file while filling up blz btable
	cdef:
		unordered_map[uint, vector[uint]] word_map
		uint doc
		unordered_map[uint, vector[uint]].iterator wm_it
		pair[uint, vector[uint]] word_docs
		uint word
		vector[uint] docs
	with open(infilename, 'rb') as infile:
		word_set.clear()
		iidx_table = blz.btable(np.empty(0, dtype='u4,u4'), expectedlen=expectedlen, rootdir=iidx_name)
		# Fill up an in-memory iidx called word_map
		for doc,line in enumerate(infile):
			line_comma_split = line.split(',')
			pre_kvs = line_comma_split[-1].split()[1:]
			for kv_str in pre_kvs:
				k,v = kv_str.split(':')
				word_map[<size_t>int(k)].push_back(doc)
		# Convert in-memory iidx to out-of-memory iidx
		wm_it = word_map.begin()
		while wm_it != word_map.end():
			word_docs = deref(inc2(wm_it))
			word = word_docs.first
			docs = word_docs.second
			for doc in docs:
				iidx_table.append((word, doc))
		word_map.clear()
		iidx_table.flush()

		# Wrap into baggage
		iidx = Baggage(iidx_table)


cdef class Baggage(object):
	'''
	`Baggage` stands for "bags," or "a collection of bags."
	`Baggage` exposes a "bag accessor" method, which essentially returns
	a slice (bag) from its underlying out-of-core, sparse-format datastore.
	To do so, `Baggage` keeps track of each bag's index and length through
	its C-allocated `lens` and `starts` arrays. Actually they're implemented
	as C++ STL vectors.
	'''

	def __cinit__(self, table):
		'''
		You can build a Baggage object by:
		A. Pass in `kvs` (dict) & `labels` (list) one-by-one
			- `starts` & `lens` will update @ ea. step
			- Option A is implemented outside of this class-scope

		B. Pass in a blz.btable with no other information
			- `starts` & `lens` will update in bulk
			>>> Baggage(blz.btable(np.empty(0)))

		C. Pass in the libsvm-format filename
			- extract_XY returns `kvs` (dict) & `labels` (list) one-by-one
			  as a generator
			- `starts` & `lens` will update @ ea. step as in option (A)
			- Option C is implemented outside of this class-scope

		*** Option B is the only allowed within this constructor method. ***

		Also, there are basically 3 types of `Baggage`s: `X`, `Y`, and `iidx`.
		These types must be hard-coded into the __init__ process. So __cinit__
		will set the static dispatch for __getitem__ (the bag accessor method).

		`dispatch` is a string that denotes which type of __getitem__ to use.

		'''
		# Option B
		if isinstance(table, blz.btable):
			self.ooc_store = table
			self.fill_starts_lens()
		else:
			raise AssertionError("Argument `table` must be an instance of `blz.btable`!")


	def fill_starts_lens(self):
		curr_len = 1
		prev_doc = self.ooc_store[0][0]
		self.starts.push_back(0)
		for start,datum in enumerate(self.ooc_store):
			curr_doc = datum[0]
			if prev_doc != curr_doc:
				prev_doc = curr_doc
				self.starts.push_back(start)
				self.lens.push_back(curr_len)
				curr_len = 0
			curr_len += 1
		self.num_words = self.starts.back() + self.lens.back()

	# We need a bag accessor
	cdef Word[:] __get_X(self, size_t x):
		'''
		Example call:
		>>> b = Baggage(some_blz_btable)
		>>> cdef Word[:] bag = b[14]

		'''
		cdef Word[:] ans = self.ooc_store[self.starts[x] : self.starts[x] + self.lens[x]]
		#return <Word[:]> self.ooc_store[self.starts[x] : self.starts[x] + self.lens[x]]
		return ans

	cdef Label[:] __get_Y(self, size_t x):
		'''
		Example call:
		>>> b = Baggage(some_blz_btable)
		>>> cdef Label[:] bag = b[14]

		'''
		cdef Label[:] ans = self.ooc_store[self.starts[x] : self.starts[x] + self.lens[x]]
		#return <Label[:]> self.ooc_store[self.starts[x] : self.starts[x] + self.lens[x]]
		return ans

	cdef Doc[:] __get_iidx(self, size_t x):
		'''
		Doc is a made-up structure that is the inverse analog of Word... will be implemented soon...
		Example call:
		>>> b = Baggage(some_blz_btable)
		>>> cdef Doc[:] bag = b[14]

		'''
		cdef Doc[:] ans = self.ooc_store[self.starts[x] : self.starts[x] + self.lens[x]]
		#return <Doc[:]> self.ooc_store[self.starts[x] : self.starts[x] + self.lens[x]]
		return ans	

	def __len__(self):
		return self.starts.size()

