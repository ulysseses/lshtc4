import time

def print_time(f):
	''' Decorate function `f` such that after f.__call__, its duration in
	time is printed to stdout '''
	def inner(*args, **kwargs):
		t0 = time.time()
		ans = f(*args, **kwargs)
		t1 = time.time()
		print "%s call duration:" % f.__name__, t1 - t0, "sec"
		return ans
	return inner
