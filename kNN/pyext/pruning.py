from __future__ import division
from collections import Counter
from itertools import izip


class LabelCounter(Counter):
    def __init__(self, corpus=None, *args, **kwargs):
        super(LabelCounter, self).__init__(*args, **kwargs)
        if corpus:
            self.build_counter(corpus)

    def build_counter(self, corpus):
        self.total_count = 0
        for labels in corpus:
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
        plt.ylabel("occurrence in corpus");

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
    def __init__(self, corpus, binary=False, *args, **kwargs):
        ''' corpus can be scipy.sparse.dok_matrix or list of {}'s '''
        super(WordCounter, self).__init__()
        self.binary = binary
        self.__build_counter(corpus)

    def __build_counter(self, corpus):
        self.total_count = 0
        if self.binary:
            for doc in corpus:
                self.total_count += len(doc)
                for word in doc.iterkeys():
                    self[word] += 1
        else:
            for doc in corpus:
                self.total_count += len(doc)
                for word,count in doc.iteritems():
                    self[word] += count
        self.d = len(self.keys())

def prune_corpora(X, Y, label_counter=None, word_counter=None):
    def f(labels):
        return [label for label in labels if label in label_counter]
    def g(doc):
        return {hash_id:tf for (hash_id, tf) in doc.iteritems()
            if hash_id in word_counter}
    
    if label_counter:
        X[:] = [doc for i,doc in enumerate(X) if f(Y[i])]
        Y[:] = [f(labels) for labels in Y if f(labels)]
    if word_counter:
        X[:] = [g(doc) for doc in X if g(doc)]




