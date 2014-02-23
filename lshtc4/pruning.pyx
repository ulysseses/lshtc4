#distutils: language=c++
#cython: boundscheck = False
#cython: wraparound = False
from __future__ import division
from collections import Counter, defaultdict
from itertools import izip

import kNN.cppext.outofcore.initialize_h5
from kNN.cppext.container cimport unordered_map, unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from lshtc4.partial_sort cimport partial_sort
from lshtc4.utils cimport comp_pair, Word, Label
from cython.operator cimport dereference as deref, preincrement as inc, \
    postincrement as inc2

import numpy as np
cimport numpy as np

ctypedef unsigned int uint


cdef class LabelCounter(object):
    cdef unordered_map[uint, uint] cmap
    cdef unordered_map[uint, uint].iterator it
    cdef size_t size
    cdef size_t d, total_count

    def __cinit__(self, object Y=None, list lst=None):
        #self.cmap = unordered_map[int, int]()
        self.it = self.cmap.begin()
        self.size = 0
        if lst:
            self.__unpack(lst)
            break
        if Y:
            self.__build_counter(Y)

    def __pack(self):
        ''' helper function to pack content '''
        cdef unordered_map[int,int].iterator it = self.blah.begin()
        cdef pair[int,int] fs
        lst = []
        while it != self.blah.end():
            fs = deref(inc2(it))
            lst.append((fs.first, fs.second))
        return lst

    def __unpack(self, lst):
        ''' helper function to unpack content '''
        cdef int f, s
        for f,s in lst:
            self.__setitem__(f, s)

    def __reduce__(self):
        lst = self.__pack()
        return (LabelCounter, (lst,))

    def __dealloc__(self):
        pass # cmap is non-heap...?

    def __cmp__(self, LabelCounter other):
        if len(self) != len(other): return 0
        cdef unordered_map[uint,uint].iterator u = self.cmap.begin()
        cdef unordered_map[uint,uint].iterator v = other.cmap.begin()
        cdef pair[uint,uint] ufs, vfs
        cdef uint uf, us, vf, vs
        while u != self.cmap.end():
            ufs = deref(inc2(u); vfs = deref(inc2(v))
            if (ufs.first != vfs.first) or (ufs.second != vfs.second):
                return 0
        return 1

    def __len__(self):
        return self.size

    def __setitem__(self, uint x, uint y):
        if self.cmap.find(x) == self.cmap.end():
            self.size += 1
            self.it = self.cmap.begin()
        self.blah[x] = y

    def __delitem__(self, uint x):
        if x in self:
            self.cmap.erase(x)
            self.size -= 1
            self.it = self.cmap.begin()
        else:
            raise KeyError('%d' % x)

    def __getitem__(self, uint x):
        if self.cmap.find(x) == self.cmap.end():
            raise KeyError('%d is not a stored key' % x)
        return self.cmap[x]

    def __contains__(self, uint x):
        cdef unordered_map[uint,uint].iterator got = self.cmap.find(x)
        return 1 if got != self.cmap.end() else 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.it != self.cmap.end():
            return deref(inc2(self.it)).second
        else:
            self.it = self.cmap.begin()
            raise StopIteration()

    def keys(self):
        cdef np.uint32_t[:] keys = np.empty(self.size, dtype=np.uint32)
        cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
        cdef size_t i = 0
        while temp != self.cmap.end():
            keys[inc2(i)] = deref(inc2(temp)).first
        return keys

    def keys2(self, np.uint32_t[:]& input):
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
        cdef np.uint32_t[:] values = np.empty(self.size, dtype=np.uint32)
        cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
        cdef size_t i = 0
        while temp != self.cmap.end():
            values[inc2(i)] = deref(inc2(temp)).second
        return values

    def values2(self, np.uint32_t[:]& input):
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
        cdef np.uint32_t[:] items = np.empty((self.size, 2), dtype=np.uint32)
        cdef unordered_map[uint, uint].iterator temp = self.cmap.begin()
        cdef pair[uint,uint] kv
        cdef size_t i = 0
        while temp != self.cmap.end():
            kv = deref(inc2(temp))
            items[i, 0] = kv.first
            items[i, 1] = kv.second
            i += 1
        return items

    def items2(self, np.uint32_t[:]& input):
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

    def __build_counter(self, Y):
        self.total_count = Y.nrows
        cdef Label l
        for r in Y:
            l = r[:]
            self.cmap[l] = 0
        for r in Y:
            l = r[:]
            self.cmap[l] += 1
        self.d = self.cmap.size()
        self.it = self.cmap.begin()

    def prune(self, uint no_below=1, double no_above=1.0, ssize_t max_n=-1):
        cdef size_t ABOVE_COUNT = no_above * self.total_count
        cdef unordered_map[uint, uint].iterator temp = self.cmap.begin()
        cdef uint label, count
        while temp != self.cmap.end():
            label, count = deref(temp)
            if count < no_below or count > ABOVE_COUNT:
                self.total_count -= count
                self.d -= 1
                temp = self.cmap.erase(temp)
            else:
                inc(temp)
        cdef size_t i = 0
        if max_n != -1 and self.size > max_n:
            self.d = max_n
            temp = self.cmap.begin()
            while i < max_n:
                i += 1
                inc(temp)
            while temp != self.cmap.end():
                self.total_size -= deref(temp).second
                temp = self.cmap.erase(temp)
        self.it = self.cmap.begin()

    def most_common(ssize_t much=-1):
        if much == -1: much = self.cmap.size()
        cdef vector[pair[uint, uint]] dfs
        dfs.reserve(self.cmap.size())
        cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
        cdef size_t i = 0
        while temp != self.cmap.end():
            dfs[inc2(i)] = deref(temp)
        partial_sort(dfs.begin(), dfs.begin()+much, dfs.end(),
            comp_pair)
        return dfs

    def most_common2(vector[pair[uint,uint]]& input, ssize_t much=-1):
        if much == -1: much = self.cmap.size()
        input.reserve(self.cmap.size())
        cdef unordered_map[uint,uint].iterator temp = self.cmap.begin()
        cdef size_t i = 0
        while temp != self.cmap.end():
            input[inc2(i)] = deref(temp)
        partial_sort(input.begin(), input.begin()+much, input.end(),
            comp_pair)

    def analyze_top_dfs(self, int most_common=100):
        cdef vector[pair[uint,uint]] dfs
        self.most_common2(dfs, much=most_common)
        for word, count in dfs:
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
    def __cinit__(self, X=None, binary=False):
        self.binary = binary
        if X:
            self.__build_counter(X)

    def __build_counter(self, X):
        cdef uint word, count
        if not self.binary:
            for r in X:
                word = r['word']
                self.cmap[word] = 0
            for r in X:
                word = r['word']
                count = r['count']
                self.cmap[word] += count
        else:
            self.total_count = 0
            for r in X:
                word = r['word']
                self.cmap[word] = 0
            for r in X:
                word = r['word']
                self.cmap[word] += 1
        self.total_count = X.nrows
        self.d = self.cmap.size()
        self.it = self.cmap.begin()

def prune_docs(X0, Y0, X1, Y1, counter):
    ''' Using `counter`, prune X0 and Y1 and store into X1 and Y1
        sample usage:

            # Assuming X, Y, pruned1X, pruned1Y, pruned2X, pruned2Y,
            #     pruned3X, & pruned3Y exist
            prune_docs(X, Y, pruned1X, pruned1Y, label_counter)
            #f.remove_node('/', 'X'); f.remove_node('/', 'Y')
            prune_docs(pruned1X, pruned1Y, pruned2X, pruned2Y, word_counter)
            #f.remove_node('/', 'pruned1X'); f.remove_node('/', 'pruned2Y')
            prune_docs(pruned2X, pruned2Y, pruned3X, pruned3Y, bin_word_counter)
            #f.remove_node('/', 'pruned3X'); f.remove_node('/', 'pruned3Y')

    '''
    def order(A0, B0, A1, B1):
        ''' A helper function that serves as a template in deciding
            which corpus (X or Y) to prune initially according to the type of
            `counter` given as an argument to `prune_docs`
        '''
        cdef unsigned int indleft = 0
        cdef unsigned int indright = 0
        cdef unordered_set[unsigned int] pruned_doc_id_set
        for r in B0:
            if <unsigned int>r['label'] in counter:
                indright += 1
            else:
                if indleft != indright:
                    B1.append(B0[indleft : indright])
                indright += 1
                indleft = indright
                # Add id of docs to be deleted within A
                pruned_doc_id_set.insert(<unsigned int>r['doc_id'])
        # test the last row since it isn't included above
        if indleft != indright:
            B1.append(B0[indleft : indright])
        B1.flush()
        # build a dict of cardinality of words for each doc, indexed by id
        cdef unordered_map[unsigned int, unsigned int] raw_doc_lens
        cdef unsigned int doc_id
        for r in A0:
            doc_id = r['doc_id']
            if raw_doc_lens.find(doc_id) == raw_doc_lens.end():
                raw_doc_lens[doc_id] = 0
            raw_doc_lens[doc_id] += 1
        # with pruned_doc_id_set, prune A0 -> A1
        indleft, indright = 0, 0
        cdef int total_size = A0.nrows
        while indright != total_size:
            doc_id = A0[indright]['doc_id']
            if pruned_doc_id_set.find(doc_id) == pruned_doc_id_set.end():
                indright += raw_doc_lens[doc_id]
            else:
                if indleft != indright:
                    A1.append(A0[indleft : indright])
                indright += 1
                indleft = indright
        # test the last row
        if indleft != indright:
            A1.append(A0[indleft : indright])
        A1.flush()

    if isinstance(counter, LabelCounter):
        # Prune Y first, then X follows suit
        order(X0, Y0, X1, Y1)
    elif isinstance(counter, WordCounter):
        # Prune X first, then Y follows suit
        order(Y0, X0, Y1, X1)
    else:
        raise AssertionError("counter must be an instance of LabelCounter" \
            "or WordCounter!")