#!/usr/bin/env ipython
from __future__ import division
#import numpy as np
import random
import scipy.sparse

N = 1
d = 1e6
n = int(1e3)

def f():
	a = scipy.sparse.rand(N,d,n/d,'csr',np.float64)
	b = scipy.sparse.rand(N,d,n/d,'csr',np.float64)
	%timeit a.dot(b.T)

def sub_g(first, second):
	ans = 0
	for k in first:
		if k in second:
			ans += first[k]*second[k]
	return ans

def g():
	global first, second
	a = [(int(random.random()*d), random.random()) for i in xrange(n)]
	b = [(int(random.random()*d), random.random()) for i in xrange(n)]
	a, b = dict(a), dict(b)
	if len(a) <= len(b):
		first, second = a, b
	else:
		first, second = b, a

	%timeit sub_g(first, second)


if __name__ == '__main__':
	print "f execution time"
	f()
	print "\n\ng execution time"
	g()