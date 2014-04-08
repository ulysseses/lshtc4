import tables as tb
from collections import defaultdict
from itertools import islice
import numpy as np

def subset(iname, oname, offset, lines):
    i, o = open(iname, 'rb'), open(oname, 'wb')
    with i,o:
        for l in islice(i, offset, offset+lines):
            o.write(l)

def extract_XY(infilename, h5name='../working/train.h5', mode='r+'):
    '''
    Given a raw libsvm multi-label formatted training text file, extract the labels
    and (hash, tf) pairs. Store them respectively into binary formats
    i.e. sparse-matrix

    line = "545, 32 8:1 18:2"
    line_comma_split = line.split(',')          # ['545', ' 32 8:1 18:2']
    labels = line_comma_split[:-1]              # ['545']
    pre_kvs = line_comma_split[-1].split()      # ['32', '8:1', '18:2']
    labels.append(pre_kvs[0])                   # ['545', '32']
    labels = [int(label) for label in labels]   ##[545, 32]
    pre_kvs = pre_kvs[1:]                       # ['8:1', '18:2']
    kvs = {}
    for kv_str in pre_kvs:
        k,v = kv_str.split(':')
        kvs[int(k)] = int(v)                    ##{8:1, 18:2}
    '''
	f = tb.openFile(h5name, mode)
	X, Y = f.root.X, f.root.Y
	rX, rY = X.row, Y.row
	with open(infilename, 'rb') as infile:
		for doc, line in enumerate(infile):
			line_comma_split = line.split(',')
			pre_kvs = line_comma_split[-1].split()
			labels.append(pre_kvs[0])
			labels = [int(label) for label in labels]
			pre_kvs = pre_kvs[1:]
			for kv_str in pre_kvs:
				k,v = kv_str.split(':')
				rX['doc'] = doc
				rX['word'] = int(k)
				rX['count'] = int(v)
				rX.append()
			X.flush()
			for label in labels:
				rY['doc'] = doc
				rY['label'] = int(label)
				rY.append()
			Y.flush()
	f.close()

def transform_tfidf(bin_word_counter, h5name='', X=None, tfidfX=None):
	''' Transform X to its modified tfidfX form '''
	if h5name:
		f = tb.openFile(h5name, mode='r+')
		X, tfidfX = f.root.pruned3X, f.root.tfidfX
    else:
        if (not X) or (not tfidfX):
            raise AssertionError('if h5name not provided, please provide' \
                'X and tfidfX manually.')
    r = tfidfX.row
    for row in X:
    	r['doc'] = row['doc']
    	r['word'] = row['word']
    	r['tfidf'] = np.log(row['count'] + 1) * log(n / \
    		bin_word_counter[row['word']])
    	r.append()
    tfidfX.flush()
    if h5name:
    	f.close()