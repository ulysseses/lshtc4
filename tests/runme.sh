#!/bin/bash

# $LSHTC4_DIR/tests/ $ ./runme.sh py_benchmark.py 0 5
# $LSHTC4_DIR/tests/ $ ./runme.sh cython_benchmark.py 0 5
# $LSHTC4_DIR/tests/ $ ./runme.sh main_cython.py 0 5

for (( i=$2; i<=$3; i++ ))
do
	if [ $i -eq $2 ]
	then
		python $1 $i | tee "$1.log"
	fi
	python $1 $i | tee -a "$1.log"
done