#distuils: language = c++

from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef api vector[int] *func():
    cdef vector[int] *v = new vector[int]()
    deref(v).push_back(3)
    return v