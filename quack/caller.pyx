from quacker import quack
import redis

cdef public void call_quack():
	print redis.__file__
	quack()