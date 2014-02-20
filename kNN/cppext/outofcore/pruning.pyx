#distutils: language=c++
#cython: boundscheck = False
#cython: wraparound = False
from __future__ import division
from collections import Counter, defaultdict
from itertools import izip
import kNN.cppext.outofcore.initialize_h5
from kNN.cppext.container cimport unordered_map, unordered_set


class LabelCounter(Counter):
    def __init__(self, Y=None, *args, **kwargs):
        super(LabelCounter, self).__init__(*args, **kwargs)
        if Y:
            self.build_counter(Y)

    def build_counter(self, Y):
        self.total_count = 0
        for labels in Y:
            self.total_count += len(labels)
            for label in labels:
                pre_count = self.__getitem__(label)
                self.__setitem__(label, pre_count+1)
        self.d = len(self.keys())

    def prune(self, no_below=1, no_above=1.0, max_n=None):
        ABOVE_COUNT = no_above*self.total_count
        remove = []
        for label, count in self.iteritems():
            if count < no_below or count > ABOVE_COUNT:
                self.total_count -= count
                self.d -= 1
                remove.append(label)
        for label in remove:
            self.__delitem__(label)
        if max_n:
            self.counter = Counter(self.counter.most_common(max_n))

    def analyze_top_dfs(self, most_common=100):
        sorted_dfs = self.most_common(most_common)
        for word, count in sorted_dfs:
            print "hash: %7d\tcount: %d\tfreq: %.3f" % \
                (word, count, count/self.total_count)

    def display_hist(self):
        import numpy as np
        from matplotlib import pyplot as plt

        def delta(a, b):
            b[0] = a[0]
            for i in xrange(1, len(a)):
                b[i] = b[i-1] + a[i]

        sorted_dfs = self.most_common()
        y = [tup[1] for tup in sorted_dfs]
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
    def __init__(self, X, binary=False, *args, **kwargs):
        ''' X can be scipy.sparse.dok_matrix or list of {}'s '''
        super(WordCounter, self).__init__()
        self.binary = binary
        self.__build_counter(X)

    def __build_counter(self, X):
        self.total_count = 0
        if not self.binary:
            for r in X:
                self.total_count += len(r)
                self[r['word']] += r['tfidf']
        else:
            for r in X:
                self.total_count += len(r)
                self[r['word']] += 1

        self.d = len(self.keys())


def prune_examples(X0, Y0, X1, Y1, counter):
    ''' Using `counter`, prune X0 and Y1 and store into X1 and Y1
        sample usage:

            # Assuming X, Y, pruned1X, pruned1Y, pruned2X, pruned2Y,
            #     pruned3X, & pruned3Y exist
            prune_examples(X, Y, pruned1X, pruned1Y, label_counter)
            #f.remove_node('/', 'X'); f.remove_node('/', 'Y')
            prune_examples(pruned1X, pruned1Y, pruned2X, pruned2Y, word_counter)
            #f.remove_node('/', 'pruned1X'); f.remove_node('/', 'pruned2Y')
            prune_examples(pruned2X, pruned2Y, pruned3X, pruned3Y, bin_word_counter)
            #f.remove_node('/', 'pruned3X'); f.remove_node('/', 'pruned3Y')

    '''
    def order(A0, B0, A1, B1):
        ''' A helper function that serves as a template in deciding
            which corpus (X or Y) to prune initially according to the type of
            `counter` given as an argument to `prune_examples`
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