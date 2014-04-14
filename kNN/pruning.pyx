#distutils: language = c++
#cython: boundscheck = True
#cython: wraparound = True
from __future__ import division

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc, \
	postincrement as inc2

from container cimport unordered_map, unordered_set
from types cimport *
from algorithm cimport partial_sort_2

import numpy as np
cimport numpy as np

ctypedef bint (*Compare)(pair[uint,uint], pair[uint,uint])
cdef inline bint comp_pair(pair[uint,uint] x, pair[uint,uint] y):
	''' A comparison function that returns 1/True or 0/False if x > y
		based on the value of the 2nd element of the pair. '''
	return x.second > y.second


cdef class LabelCounter(object):
	''' LabelCounter is an unordered_map implementation of Python's
		collections.Container class. It can pickle/unpickle as well.
		Additional features include pruning and histogram plotting. '''

	def __cinit__(self, kwarg1=None):
		'''
		There are 4 ways of initializing the counter.
		a. unpacking from a pickled file
		b. constructing from the libsvm file
		c. initializing as a default, empty counter
		d. initializing LabelCounter & WordCounter together in an
		external function

		Option (d) is implemented below.
		'''
		self.total_count = 0
		self.it = self.cmap.begin()
		if type(kwarg1) == list:
			self.__unpack(kwarg1)
		elif type(kwarg1) == str:
			self.__build_counter(kwarg1)

	def __dealloc__(self):
		pass

	def __pack(self):
		''' Helper function to pack content for pickling '''
		cdef pair[uint, uint] fs # first, second
		lst = []
		while it != self.cmap.end():
			fs = deref(inc2(it))
			lst.append((fs.first, fs.second))
		self.it = self.cmap.begin()
		return lst

	def __unpack(self, lst):
		''' Helper function to unpack content for un-pickling '''
		# modified, faster version of __setitem__
		self.total_count = 0
		cdef uint f, s
		for f,s in lst:
			self.total_count += s
			self.cmap[f] = s
		self.it = self.cmap.begin()

	def __reduce__(self):
		lst = self.__pack()
		return (LabelCounter, (lst,))

	def __build_counter(self, infilename):
		'''
		Read the contents of the file named `infilename` and tally up the
		label counts.
		'''
		self.total_count = 0
		with open(infilename, 'rb') as infile:
			for line in infile:
				line_comma_split = line.split(',')
				labels = line_comma_split[:-1]
				for label in labels:
					self.cmap[<uint>int(label)] += 1
					self.total_count += 1
			self.it = self.cmap.begin()

	def __cmp__(self, other):
		if len(self) != len(other): return 0
		cdef unordered_map[uint,uint].iterator u = self.cmap.begin()
		cdef unordered_map[uint,uint].iterator v = other.cmap.begin()
		cdef pair[uint,uint] ufs, vfs
		cdef uint uf, us, vf, vs
		while u != self.cmap.end():
			ufs = deref(inc2(u)); vfs = deref(inc2(v))
			if (ufs.first != vfs.first) or (ufs.second != vfs.second):
				return 0
		return 1

	def __len__(self):
		return self.cmap.size()

	def __setitem__(self, uint x, uint y):
		self.cmap[x] = y
		self.it = self.cmap.begin()

	def __contains__(self, uint x):
		return 1 if self.cmap.find(x) != self.cmap.end() else 0

	def __delitem__(self, uint x):
		if x in self:
			self.cmap.erase(x)
			self.it = self.cmap.begin()
		# else:
		# 	raise KeyError('%d' % x)
		# ^ comment out above for speedup

	def __getitem__(self, uint x):
		# if self.cmap.find(x) == self.cmap.end():
		# 	raise KeyError('%d is not a stored key' % x)
		# ^ comment out above for speedup
		return self.cmap[x]

	def __iter__(self):
		return self

	def __next__(self):
		if self.it != self.cmap.end():
			return deref(inc2(self.it)).second
		else:
			self.it = self.cmap.begin()
			raise StopIteration()

	def keys(self):
		''' Return a cython-memoryview of the dict keys. '''
		cdef uint[:] keys = np.empty(len(self), dtype=np.uint32)
		cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
		cdef size_t i = 0
		while temp != self.cmap.end():
			keys[inc2(i)] = deref(inc2(temp)).first
		return keys

	def keys2(self, uint[:]& input):
		''' Fill a cython-memoryview with the dict keys. '''
		cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
		cdef size_t i = 0
		while temp != self.cmap.end():
			input[inc2(i)] = deref(inc2(temp)).first

	def iterkeys(self):
		cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
		while temp != self.cmap.end():
			yield deref(temp).first
			inc(temp)

	def values(self):
		''' Return a cython-memoryview of the dict vals '''
		cdef uint[:] values = np.empty(self.size, dtype=np.uint32)
		cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
		cdef size_t i = 0
		while temp != self.cmap.end():
			values[inc2(i)] = deref(inc2(temp)).second
		return values

	def values2(self, uint[:]& input):
		''' Fill a cython-memoryview with the dict vals '''
		cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
		cdef size_t i = 0
		while temp != self.cmap.end():
			input[inc2(i)] = deref(inc2(temp)).second

	def itervalues(self):
		cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
		while temp != self.cmap.end():
			yield deref(temp).second
			inc(temp)

	def items(self):
		''' Return a cython-memoryview of the dict items '''
		cdef uint[:] items = np.empty((self.size, 2), dtype=np.uint32)
		cdef unordered_map[uint, uint].iterator temp = self.cmap.begin()
		cdef pair[uint,uint] kv
		cdef size_t i = 0
		while temp != self.cmap.end():
			kv = deref(inc2(temp))
			items[i, 0] = kv.first
			items[i, 1] = kv.second
			i += 1
		return items

	def items2(self, uint[:]& input):
		''' Fill a cython-memoryview with the dict items '''
		cdef unordered_map[uint, uint].iterator temp = self.cmap.begin()
		cdef pair[uint,uint] kv
		cdef size_t i = 0
		while temp != self.cmap.end():
			kv = deref(inc2(temp))
			input[i, 0] = kv.first
			input[i, 1] = kv.second
			i += 1

	def iteritems(self):
		cdef unordered_map[uint, uint].iterator temp = self.cmap.begin()
		while temp != self.cmap.end():
			yield deref(temp)
			inc(temp)

	def prune(self, uint no_below=1, double no_above=1.0, max_n=-1):
		cdef uint ABOVE_COUNT = no_above * self.total_count
		cdef uint label, count
		cdef pair[uint,uint] temp
		while self.it != self.cmap.end():
			temp = deref(self.it)
			label = temp.first
			count = temp.second
			if count < no_below or count > ABOVE_COUNT:
				self.total_count -= count
				self.it = self.cmap.erase(self.it)
			else:
				inc(self.it)
		cdef uint i = 0
		if max_n != -1 and self.size > max_n:
			self.it = self.cmap.begin()
			while i < max_n:
				i += 1
				inc(self.it)
			while self.it != self.cmap.end():
				self.total_count -= deref(self.it).second
				self.it = self.cmap.erase(self.it)
		self.it = self.cmap.begin()

	def most_common(much=-1):
		cdef size_t cmuch
		if much <= 0: cmuch = self.cmap.size()
		else: cmuch = much
		cdef vector[pair[uint, uint]] dfs
		dfs.reserve(self.cmap.size())
		cdef size_t i = 0
		while self.it != self.cmap.end():
			dfs[inc2(i)] = deref(temp)
		partial_sort_2(dfs.begin(), dfs.begin()+cmuch, dfs.end(),
			comp_pair)
		self.it = self.cmap.begin()
		return dfs

	def most_common2(vector[pair[uint,uint]]& input, much=-1):
		cdef size_t cmuch
		if much <= 0: cmuch = self.cmap.size()
		else: cmuch = much
		input.reserve(self.cmap.size())
		cdef size_t i = 0
		while self.it != self.cmap.end():
			input[inc2(i)] = deref(temp)
		partial_sort_2(input.begin(), input.begin()+cmuch, input.end(),
			comp_pair)
		self.it = self.cmap.begin()

	def analyze_top_dfs(self, most_common=100):
		cdef vector[pair[uint,uint]] dfs
		self.most_common2(dfs, much=most_common)
		cdef size_t i
		cdef pair[uint,uint] wc
		for i in xrange(dfs.size()):
			wc = dfs[i]
			word = wc.first
			count = wc.count
			print "hash: %7d\tcount: %d\tfreq: %.3f" % \
				(word, count, count/self.total_count)

	def display_hist(self):
		from matplotlib import pyplot as plt

		def delta(a, b):
			b[0] = a[0]
			for i in xrange(1, len(a)):
				b[i] = b[i-1] + a[i]

		cdef vector[pair[uint,uint]] dfs
		self.most_common2(dfs)
		y = [tup[1] for tup in dfs]
		x = np.arange(len(y))

		plt.figure(figsize=(8,5));
		plt.loglog(x, y);
		plt.grid();
		plt.xlabel("word rank");
		plt.ylabel("occurrence in Y");

		cdf = np.empty(len(y))
		delta(y, cdf)
		cdf /= np.max(cdf) # normalize

		x50 = x[cdf > 0.50][0]
		x80 = x[cdf > 0.80][0]
		x90 = x[cdf > 0.90][0]
		x95 = x[cdf > 0.95][0]
		
		print "50%\t", x50
		print "80%\t", x80
		print "90%\t", x90
		print "95%\t", x95
		
		plt.axvline(x50, color='c');
		plt.axvline(x80, color='g');
		plt.axvline(x90, color='r');
		plt.axvline(x95, color='k');
		plt.show();  


class WordCounter(LabelCounter):
	def __cinit__(self, kwarg2=None, binary=False):
		self.binary = binary
		if type(kwarg2) == list:
			self.__unpack(kwarg2)
		elif type(kwarg2) == str:
			self.__build_counter(kwarg2)

	def __build_counter(self, infilename):
		'''
		Read the contents of the file named `infilename` and tally up the
		word counts.
		'''
		with open(infilename, 'rb') as infile:
			self.total_count = 0
			for line in infile:
				line_comma_split = line.split(',')
				pre_kvs = line_comma_split[-1].split()[1:]
				for kv_str in pre_kvs:
					k,v = kv_str.split(':')
					self.cmap[k] += 1
					self.total_count += 1
		self.it = self.cmap.begin()


def prune_docs(inname, outname, counter):
	''' Using `counter`, prune the in-file and write into the out-file
		Given the type of the `counter`, decide whether to prune `X` or `Y` first. '''

	A, B = open(inname, 'rb'), open(outname, 'wb')
	with A, B:
		if isinstance(counter, LabelCounter):
			for line in A:
				# Get labels
				line_comma_split = line.split(',')
				labels = line_comma_split[:-1]
				labels.append(line_comma_split[-1].split()[0])
				# Test membership
				filtered_labels = [label for label in labels if int(label) in counter]
				filtered_labels_str = ", ".join(filtered_labels)
				# Get kvs
				if filtered_labels:
					pre_kvs = line_comma_split[-1].split()[1:]
				# Write to B back in libsvm format: "label1, label2 word1:tf1 word2:tf2"
				B.write(filtered_labels_str + " " + " ".join(pre_kvs) + '\n')
		elif isinstance(counter, WordCounter):
			for line in A:
				# Get kvs
				line_comma_split = line.split(',')
				pre_kvs = line_comma_split[-1].split()[1:]
				kvs = {}
				for kv_str in pre_kvs:
					k,v = kv_str.split(':')
					kvs[k] = v
				# Test membership
				filtered_kvs = {k:v for (k,v) in kvs.iteritems() if int(k) in counter}
				filtered_kvs_str = " ".join(("%s:%s" for (k,v) in filtered_kvs.iteritems()))
				# Get labels
				if filtered_kvs:
					labels = line_comma_split[:-1]
					labels.append(line_comma_split[-1].split()[0])
				# Write to B back in libsvm format: "label1, label2 word1:tf1 word2:tf2"
				B.write(", ".join(labels) + " " + filtered_kvs_str + '\n')
		else:
			raise AssertionError("counter must be an instance of LabelCounter" \
				" or WordCounter!")
