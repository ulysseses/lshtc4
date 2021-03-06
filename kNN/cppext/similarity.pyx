#distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False
from __future__ import division
cimport cython
from libc.math cimport log
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc
from kNN.cppext.container cimport unordered_map, unordered_set

ctypedef unordered_map[int, int] iidict
ctypedef unordered_map[int, unordered_set[int]] isdict
ctypedef unordered_map[int, double] iddict
ctypedef unordered_map[int, double].iterator iddictitr
ctypedef unordered_set[int] iset
ctypedef unordered_set[int].iterator isetitr
ctypedef vector[unordered_map[int,double]] mapvect
ctypedef pair[int,double] idpair
ctypedef vector[vector[int]] vectvect
ctypedef unordered_map[int, vector[double]] vectmap
ctypedef unordered_map[int, vector[double]].iterator vectmapitr
ctypedef unordered_map[int, unordered_set[int]] isetdict
ctypedef unordered_map[int, unordered_set[int]].iterator isetdictitr

cdef void transform_tfidf(mapvect& corpus, iidict& bin_word_counter):
    ''' Transform corpus into modified tf-idf representation.
        t_X = transform_corpus(corpus, bin_word_counter)

        Use to transform both the test and train set.'''
    # cdef int n, w
    # cdef dict doc
    
    # n = len(corpus)
    # for doc in corpus:
    #     for w in doc:
    #         doc[w] = log(doc[w] + 1) * log(n / bin_word_counter[w])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    cdef int n, i, j
    cdef iddict doc
    cdef iddictitr it
    cdef idpair kv
    n = <int> corpus.size()
    for i in xrange(n):
        doc = corpus[i]
        it = doc.begin()
        while it != doc.end():
            kv = deref(it)
            doc[kv.first] = log(kv.second + 1) * log(<double>n / \
                <double>bin_word_counter[kv.first])
            inc(it)

cdef double norm(iddict& doc):
    cdef iddictitr it = doc.begin()
    cdef idpair hash_tf
    cdef double v
    cdef double ansatz = 0
    while it != doc.end():
        v = deref(it).second
        inc(it)
        ansatz += v*v
    return log(ansatz ** 0.5)

ctypedef bint (*compare)(idpair, idpair)
ctypedef vector[idpair].iterator pairvectitr
cdef extern from "<algorithm>" namespace "std":
    void partial_sort(pairvectitr&, pairvectitr&, pairvectitr&, compare&) nogil except +
    void partial_sort(pairvectitr&, pairvectitr&, pairvectitr&) nogil except +

cdef inline bint comp_func(idpair& x, idpair& y):
    ''' A comparison func. that returns 1/True or 0/False if x > y
        based on the value of the second element in the pair, respectively. '''
    return <bint> x.second > y.second

cdef pair[vectmap, vectmap] cossim(iddict& d_i, mapvect& t_X, int k, vectvect& t_Y, isdict& parents_index,
        isdict& children_index):
    ''' For the query `d_i`, calculate its similarity with all docs in corpus `t_X`.
        Obtain the top k-NN (i.e. k highest similar docs)
        Return a container of labels, and corresponding scores, of `d_i`'s
        most similar docs.
        Also return a container of labels, and corresponding pscores, of `d_i`.
    '''
    cdef vector[idpair] doc_scores
    cdef vectmap cat_scores_dict
    cdef int t_X_size = <int>t_X.size()
    cdef int i, doc_size, j, label
    cdef int d_i_size = <int> d_i.size()
    cdef int overlap_count = 0
    cdef double threshold = 0.3
    cdef int smaller_size
    cdef iddict doc
    cdef vector[int] labels
    cdef double top, bottom, score
    cdef iddictitr it
    cdef idpair kv
    cdef iddictitr got
    cdef pairvectitr it2
    cdef vectmap scores, pscores
    cdef vectmapitr got2
    cdef unordered_set[int] parents_set
    cdef unordered_set[int].iterator it3
    cdef int parent
    cdef unordered_set[int] children_set
    for i in xrange(t_X_size):
        doc = t_X[i]
        labels = t_Y[i]
        # Calculate numerator efficiently
        doc_size = <int> doc.size()
        top = 0
        if doc_size <= d_i_size:
            smaller_size = doc_size
            it = doc.begin()
            while it != doc.end():
                kv = deref(it)
                inc(it)
                got = d_i.find(kv.first)
                if got != d_i.end():
                    overlap_count += 1
                    top += kv.second * deref(got).second
        else:
            smaller_size = d_i_size
            it = d_i.begin()
            while it != d_i.end():
                kv = deref(it)
                inc(it)
                got = doc.find(kv.first)
                if got != doc.end():
                    overlap_count += 1
                    top += kv.second * deref(got).second
        # Calculate denominator efficiently
        # Set a magic-number threshold to skip if top is small
        if overlap_count > <int>(threshold * smaller_size):
            bottom = norm(d_i) * norm(doc)
            score = top / bottom
        else:
            score = 0
        for j in xrange(<int>labels.size()):
            label = labels[j]
            doc_scores.push_back(idpair(label,score))
            got2 = cat_scores_dict.find(label)
            if got2 == cat_scores_dict.end():
                cat_scores_dict[label] = vector[double]()
            cat_scores_dict[label].push_back(score)
    # Return the k-NN (aka top-k similar examples)
    partial_sort(doc_scores.begin(), doc_scores.begin()+k, doc_scores.end(),
        comp_func)
    # optimized/transformed scores & pscores
    it2 = doc_scores.begin()
    for i in xrange(k):
        kv = deref(it2)
        label = kv.first
        score = kv.second
        inc(it2)
        got2 = scores.find(label)
        if got2 == scores.end():
            scores[label] = vector[double]()
        scores[label].push_back(score)
        parents_set = parents_index[label]
        it3 = parents_set.begin()
        while it3 != parents_set.end():
            parent = deref(it3)
            inc(it3)
            children_set = children_index[parent]
            got2 = pscores.find(label)
            if got2 == pscores.end():
                pscores[label] = vector[double]()
            pscores[label].push_back(<double>children_set.size())
    return pair[vectmap, vectmap](scores, pscores)

cdef pair[vectmap, vectmap] cossim2(iddict& d_i, mapvect& t_X, int k, vectvect& t_Y, isdict& parents_index,
        isdict& children_index, isetdict& iidx):
    # Find all candidate doc numbers
    cdef iddictitr it
    cdef int word
    cdef isetdictitr got
    cdef iset dns
    cdef isetitr it2
    cdef iset doc_nums
    it = d_i.begin()
    while it != d_i.end():
        word = deref(it).first
        got = iidx.find(word)
        inc(it)
        if got != iidx.end():
            dns = deref(got).second
            it2 = dns.begin()
            while it2 != dns.begin():
                doc_nums.insert(deref(it2))
                inc(it2)
    # Access only the candidate docs index by doc_nums
    cdef int doc_num
    cdef iddict doc
    cdef vector[int] labels
    cdef int d_i_size
    cdef int doc_size
    cdef double top
    cdef idpair kv
    cdef iddictitr got2
    cdef bottom, score
    cdef int j
    cdef int label
    cdef vector[pair[int,double]] doc_scores
    cdef vectmap cat_scores_dict
    cdef vectmapitr got3

    d_i_size = d_i.size()
    it2 = doc_nums.begin()
    while it2 != doc_nums.end():
        doc_num = deref(it2)
        doc = t_X[doc_num]
        inc(it2)
        labels = t_Y[doc_num]
        # Calculate numerator efficiently
        doc_size = doc.size()
        top = 0
        if doc_size <= d_i_size:
            it = doc.begin()
            while it != doc.end():
                kv = deref(it)
                inc(it)
                got2 = d_i.find(kv.first)
                if got2 != d_i.end():
                    top += kv.second * deref(got2).second
        else:
            it = d_i.begin()
            while it != d_i.end():
                kv = deref(it)
                inc(it)
                got2 = doc.find(kv.first)
                if got2 != doc.end():
                    top += kv.second * deref(got2).second
        # Calculate denominator
        if top > 0:
            bottom = norm(d_i) * norm(doc)
            score = top / bottom
        else:
            score = 0
        # Push into cat_scores_dict & doc_scores
        for j in xrange(labels.size()):
            label = labels[j]
            doc_scores.push_back(idpair(label, score))
            got3 = cat_scores_dict.find(label)
            if got3 == cat_scores_dict.end():
                cat_scores_dict[label] = vector[double]()
            cat_scores_dict[label].push_back(score)
    # Return the k-NN (aka top-k similar examples)
    cdef int kk
    if k < doc_scores.size():
        kk = k
        partial_sort(doc_scores.begin(), doc_scores.begin()+k, doc_scores.end(),
            comp_func)
    else:
        kk = doc_scores.size()
    # optimized/transformed scores & pscores
    cdef pairvectitr it3
    cdef int i
    cdef vectmap scores
    cdef iset parents_set
    cdef isetitr it4
    cdef int parent
    cdef iset children_set
    cdef vectmap pscores

    it3 = doc_scores.begin()
    for i in xrange(kk):
        kv = deref(it3)
        label = kv.first
        score = kv.second
        inc(it3)
        got3 = scores.find(label)
        if got3 == scores.end():
            scores[label] = vector[double]()
        scores[label].push_back(score)
        parents_set = parents_index[label]
        it4 = parents_set.begin()
        while it4 != parents_set.end():
            parent = deref(it4)
            inc(it4)
            children_set = children_index[parent]
            got3 = pscores.find(label)
            if got3 == pscores.end():
                pscores[label] = vector[double]()
            pscores[label].push_back(children_set.size())
    return pair[vectmap, vectmap](scores, pscores)

ctypedef vector[double].iterator dvectitr
cdef extern from "<algorithm>" namespace "std":
    dvectitr max_element(dvectitr&, dvectitr&) nogil except +
    
cdef double custom_max(iddict& lut):
    ''' personal func that returns the max value inside an
        unordered_map[int,double] '''
    cdef iddictitr it = lut.begin()
    cdef double ansatz = deref(it).second
    cdef double val
    while it != lut.end():
        val = deref(it).second
        inc(it)
        if val > ansatz:
            ansatz = val
    return ansatz

#num = cython.fused_type(cython.int, cython.float, cython.double)
cdef double csum(vector[double]& vect):
    cdef double ansatz = 0
    cdef vector[double].iterator it = vect.begin()
    while it != vect.end():
        ansatz += deref(it)
        inc(it)
    return ansatz

cdef iddict optimized_ranks(vectmap& scores, vectmap& pscores, iidict& label_counter,
        double w1, double w2, double w3, double w4):
    ''' w1..w4 are weights corresponding to x1..x4 '''
    # cdef int c
    # cdef double x1, x2, x3, x4
    # ranks_dict = {}
    # for c in scores:
    #     # x1 = np.log(max(scores[c]))
    #     # x2 = np.log(sum(pscores[c]))
    #     # x3 = np.log(sum(scores[c]))
    #     # x4 = np.log(len(scores[c])/label_counter[c])
    #     x1 = max(scores[c])
    #     x2 = sum(pscores[c])
    #     x3 = sum(scores[c])
    #     x4 = len(scores[c])/label_counter[c]
    #     ranks_dict[c] = w1*x1 + w2*x2 + w3*x3 + w4*x4
    # return ranks_dict
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    cdef int c
    cdef vectmapitr it
    cdef pair[int,vector[double]] kv
    cdef vector[double] inner_scores, inner_pscores
    cdef double x1, x2, x3, x4
    cdef iddict ranks_dict
    it = scores.begin()
    while it != scores.end():
        kv = deref(it)
        inc(it)
        c = kv.first
        inner_scores = kv.second
        inner_pscores = pscores[c]
        x1 = deref(max_element(inner_scores.begin(), inner_scores.end()))
        x2 = csum(inner_pscores)
        x3 = csum(inner_scores)
        x4 = (<double>inner_scores.size())/(<double>label_counter[c])
        ranks_dict[c] = w1*x1 + w2*x2 + w3*x3 + w4*x4
    return ranks_dict

cdef vector[int] predict(iddict& ranks, double alpha):
    ''' Return a list of labels if their corresponding ranks are higher
        than a threshold provided by `alpha`.
    '''
    # cdef double max_rank, rank
    # cdef int label
    # max_rank = max(ranks.itervalues())
    # return [label for label, rank in ranks.iteritems() if rank/max_rank > alpha]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    cdef vector[int] ans
    cdef double max_rank = custom_max(ranks)
    cdef iddictitr it = ranks.begin()
    cdef idpair kv
    while it != ranks.end():
        kv = deref(it)
        inc(it)
        if kv.second / max_rank > alpha:
            ans.push_back(kv.first)
    return ans