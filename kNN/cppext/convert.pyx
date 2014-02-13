#distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False
''' Convert Pythonic containers to Cythonic containers. '''

# from unordered_map cimport unordered_map
# from unordered_set cimport unordered_set
# from libcpp.vector cimport vector
# from cython.operator cimport dereference as deref, preincrement as inc
# from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference as deref, preincrement as inc
from kNN.cppext.container cimport unordered_map, unordered_set

cdef vector[unordered_map[int, double]] cythonize_X(object X):
    ''' Convert list(dict(int:double)) to vector[unordered_map[int,double]] '''
    cdef vector[unordered_map[int, double]] cX
    cdef unordered_map[int,double]* cdoc
    cdef int i
    for i,doc in enumerate(X):
        cX.push_back(unordered_map[int,double]())
        cdoc = &cX[i]
        for k,v in doc.iteritems():
            deref(cdoc)[<int>k] = <double>v
    return cX

cdef vector[vector[int]] cythonize_Y(object Y):
    ''' Convert list(list(int)) to vector[vector[int]] '''
    cdef vector[vector[int]] cY
    cdef vector[int]* clabels
    cdef int i
    for i,labels in enumerate(Y):
        cY.push_back(vector[int]())
        clabels = &cY[i]
        for label in labels:
            clabels.push_back(<int>label)
    return cY

cdef unordered_map[int, unordered_set[int]] cythonize_index(object index):
    ''' Convert dict(int:set(int)) to unordered_map[int, unordered_set[int]] '''
    cdef unordered_map[int, unordered_set[int]] cindex
    cdef unordered_set[int]* cnodes
    cdef int ind
    for ind, nodes in index.iteritems():
        cindex[ind] = unordered_set[int]()
        cnodes = &cindex[ind]
        for node in nodes:
            cnodes.insert(<int>node)
    return cindex

cdef unordered_map[int,int] cythonize_counter(object counter):
    ''' Convert dict(int:int) to unordered_map[int,int] '''
    cdef unordered_map[int,int] ccounter
    for token, count in counter.iteritems():
        ccounter[<int>token] = <int>count
    return ccounter

