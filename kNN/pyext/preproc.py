from collections import defaultdict
from itertools import islice

def subset(iname, oname, offset, lines):
	i, o = open(iname, 'rb'), open(oname, 'wb')
	with i,o:
		for l in islice(i, offset, offset+lines):
			o.write(l)

def extract_XY(infilename):
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

	X, Y = [], []
	with open(infilename, 'rb') as f:
		for line in f:
			line_comma_split = line.split(',')
			labels = line_comma_split[:-1]
			pre_kvs = line_comma_split[-1].split()
			labels.append(pre_kvs[0])
			labels = [int(label) for label in labels]
			kvs = pre_kvs[1:]
			doc = {}
			for kv_str in kvs:
				k,v = kv_str.split(':')
				doc[int(k)] = int(v)
			X.append(doc)
			Y.append(labels)

	return X, Y

def extract_Y(infilename):
	Y = []
	with open(infilename, 'rb') as f:
		for line in f:
			line_comma_split = line.split(',')
			labels = line_comma_split[:-1]
			pre_kvs = line_comma_split[-1].split()
			labels.append(pre_kvs[0])
			labels = [int(label) for label in labels]
			Y.append(labels)
	return Y


def extract_parents(Y, infilename):
	''' Extract the immediate parents_index of each leaf node.
		Builds an index of child->parents_index
		`parents_index` is a dict of sets
	'''
	parents_index = {int(label):set([]) for labels in Y for label in labels}
	with open(infilename, 'rb') as f:
		for line in f:
			# guaranteed ea. line has 2 tokens
			parent, child = [int(x) for x in line.split()]
			if child in parents_index:
				parents_index[child].add(parent)
	return parents_index

def invert_index(parents_index):
	''' Build an inverse index of parent->children.
		Focus our attention only on immediate parents_index of leaf nodes.
		No grandparents_index (of leaf nodes) allowed. '''
	children_index = defaultdict(set)
	for child, p_set in parents_index.iteritems():
		for p in p_set:
			children_index[p].add(child)
	return children_index

def inverse_index(X):
	''' Create a word->doc# index '''
	iidx = defaultdict(set)
	for i,doc in enumerate(X):
		for word in doc:
			iidx[word].add(i)
	return iidx



