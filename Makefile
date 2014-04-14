cython_libraries:
	python setup.py build_ext --inplace
	mkdir clib/
	mkdir cython_libraries/
	mv pkg/*.cpp clib/
	mv pkg/*.so cython_libraries/

clean:
	rm -rf clib/
	rm -rf cython_libraries/
	rm -rf build/
