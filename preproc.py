
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