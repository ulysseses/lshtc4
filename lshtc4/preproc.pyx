#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
import tables as tb
from collections import defaultdict
from itertools import islice

from kNN.cppext.container cimport unordered_map, unordered_set
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stdio cimport fopen, fclose, getline
from cython.operator cimport dereference as deref, preincrement as inc

ctypedef unsigned int uint

def subset(iname, oname, offset, lines):
	i, o = open(iname, 'rb'), open(oname, 'wb')
	with i,o:
		for l in islice(i, offset, offset+lines):
			o.write(l)

def extract_XY(infilename, h5name="../working/train.h5", mode='r+'):
	'''
	Given a raw libsvm multi-label formatted training text file, extract the labels
	and (hash, tf) pairs. Store them respectively into binary formats
	i.e. sparse-matrix

	line = "545, 32 8:1 18:2"
	line_comma_split = line.split(',')			# ['545', ' 32 8:1 18:2']
	labels = line_comma_split[:-1]				# ['545']
	pre_kvs = line_comma_split[-1].split()  	# ['32', '8:1', '18:2']
	labels.append(pre_kvs[0])					# ['545', '32']
	labels = [int(label) for label in labels]	##[545, 32]
	pre_kvs = pre_kvs[1:]						# ['8:1', '18:2']
	kvs = {}
	for kv_str in pre_kvs:
		k,v = kv_str.split(':')
		kvs[int(k)] = int(v)					##{8:1, 18:2}
	'''
	f = tb.openFile(h5name, 'r+')
	X = f.root.X
	Y = f.root.Y
	rX, rY = X.row, Y.row
	cdef int doc = 0

	cdef FILE* cfile
	cfile = fopen(infilename, 'rb')
	if cfile == NULL:
		raise FileNotFoundError(2, "No such file: '%s'" % infilename)

	cdef char* line = NULL
	cdef size_t l = 0
	cdef ssize_t read

	while True:
		read = getline(&line, &l, cfile)
		if read == -1: break
		line_comma_split = line.split(',')
		pre_kvs = line_comma_split[-1].split()
		labels.append(pre_kvs[0])
		labels = [<uint>int(label) for label in labels]
		pre_kvs = pre_kvs[1:]
		for kv_str in pre_kvs:
			k,v = (<char*>kv_str).split(':')
			rX['doc'] = doc
			rX['word'] = <uint>int(k)
			rX['count'] = <uint>int(v)
			rX.append()
		for label in labels:
			rY['doc'] = doc
			rY['label'] = <uint> int(label)
			rY.append()
		doc += 1
	fclose(cfile)
	X.flush(); Y.flush()
	f.close()


cdef void extract_parents(object& Y, char* infilename,
		unordered_map[uint, unordered_set[uint]]& parents_index):
	''' Extract the immediate parents_index of each leaf node.
		Builds an index of child->parents_index
		`parents_index` is a dict of sets
	'''
	cdef list labels
	for labels in Y:
		for label in labels:
			parents_index[label] = unordered_set[int]()
	cdef char* line
	cdef uint parent, child
	with open(infilename, 'rb') as f:
		for line in f:
			# guaranteed ea. line has 2 tokens
			parent, child = [(<uint>int(x)) for x in line.split()]
			if child in parents_index:
				parents_index[child].insert(parent)


def parents2children(unordered_map[uint, unordered_set[uint]]& parents_index,
		unordered_map[uint, unordered_set[uint]]& children_index):
	''' Build an inverse index of parent->children.
		Focus our attention only on immediate parents_index of leaf nodes.
		No grandparents_index (of leaf nodes) allowed. '''
	cdef unordered_map[uint, unordered_set[uint]].iterator it = parents_index.begin()
	cdef pair[uint, unordered_set[uint]] kv
	cdef unordered_set[uint] p_set
	cdef unordered_set[uint].iterator it2
	while it != parents_index.end():
		kv = deref(it)
		p_set = kv.second
		inc(it)
		it2 = p_set.begin()
		while it2 != p_set.end():
			children_index[deref(it2)].insert(kv.first)


cdef void inverse_index(object& X, unordered_map[uint, unordered_set[uint]]& iidx):
	cdef uint word
	for r in X:
		word = r['word']
		if iidx.find(word) == iidx.end():
			iidx[word] = unordered_set[int]()
		iidx[word].insert(<uint>r['doc'])

cdef void get_doc_lens(object& corpus, vector[uint]& doc_len_idx):
	""" Create a doc_len_idx from corpus (X or Y) """
	cdef uint doc
	cdef uint curr_doc = 0
	cdef uint doc_len = 0
	for r in corpus:
		doc = r['doc']
		if doc == curr_doc:
			doc_len += 1
		else:
			while doc > doc_len_idx.size():
				doc_len_idx.push_back(0)
			doc_len_idx.push_back(doc_len)
			doc_len = 0
			curr_doc = doc

cdef void get_doc_starts(vector[uint]& doc_len_idx, vector[uint]& doc_start_idx):
	''' Convert doc_len_idx to doc_start_idx '''
	cdef uint i
	cdef uint curr_sum = 0
	doc_start_idx.push_back(0)
	for i in xrange(doc_len_idx.size()):
		curr_sum += doc_len_idx[i]
		doc_start_idx.push_back(curr_sum)


