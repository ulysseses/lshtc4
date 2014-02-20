#distutils: language = c++
#cython: boundscheck = False
#cython: wraparound = False
import tables as tb
from collections import defaultdict
from itertools import islice

from kNN.cppext.container cimport unordered_map, unordered_set
from libcpp.vector cimport vector
from libc.stdio cimport fopen, fclose, getline
from cython.operator cimport dereference as deref, preincrement as inc

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
	cdef int doc_id = 0

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
		labels = [<unsigned int>int(label) for label in labels]
		pre_kvs = pre_kvs[1:]
		for kv_str in pre_kvs:
			k,v = (<char*>kv_str).split(':')
			rX['doc_id'] = doc_id
			rX['word'] = <unsigned int>int(k)
			rX['count'] = <unsigned int>int(v)
			rX.append()
		for label in labels:
			rY['doc_id'] = doc_id
			rY['label'] = <unsigned int> int(label)
			rY.append()
		doc_id += 1
	fclose(cfile)
	X.flush(); Y.flush()
	f.close()


def extract_parents(Y, infilename):
	''' Extract the immediate parents_index of each leaf node.
		Builds an index of child->parents_index
		`parents_index` is a dict of sets
	'''
	cdef char* line
	cdef unsigned int parent, child
	parents_index = {(<unsigned int>int(label)):set([]) for labels in Y 
		for label in labels}
	with open(infilename, 'rb') as f:
		for line in f:
			# guaranteed ea. line has 2 tokens
			parent, child = [(<unsigned int>int(x)) for x in line.split()]
			if child in parents_index:
				parents_index[child].add(parent)
	return parents_index

def invert_index(parents_index):
	''' Build an inverse index of parent->children.
		Focus our attention only on immediate parents_index of leaf nodes.
		No grandparents_index (of leaf nodes) allowed. '''
	cdef unsigned int child, p
	children_index = defaultdict(set)
	for child, p_set in parents_index.iteritems():
		for p in p_set:
			children_index[p].add(child)
	return children_index

def inverse_index(X):
	''' Create a word->doc# index '''
	iidx = defaultdict(set)
	for r in X:
		iidx[r['word']].add(r['doc_id'])
	return iidx
