#!/usr/bin/env python
''' Usage

manually:
	python main_cython.py 0 # <-- cb.stage0()
	python main_cython.py 3 # <-- cb.stage3()

automated .sh script:
	./runme.sh main_cython.py 0 5 <-- runs cb.stage0 .. loaded_main

'''
import sys
sys.path.insert(0, '..')

from kNN.cppext.tests import cython_benchmark as cb

if __name__ == '__main__':
	if len(sys.argv) == 1:
		raise NotImplementedError("One-go not yet implemented."
			" Try cb.stage*() for now.")
		#cb.onego_main()
	else:
		stage = int(sys.argv[1])
		if stage == 0: cb.stage0()
		if stage == 1: cb.stage1()
		if stage == 2: cb.stage2()
		if stage == 3: cb.stage3()
		if stage == 4: cb.stage4()
		if stage == 5: cb.stage5()
		if stage == 6: cb.loaded_main()