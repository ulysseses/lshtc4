#!/bin/bash

# $LSHTC4_DIR/tests/ # ./runme.sh py_benchmark 0 5
# $LSHTC4_DIR/tests/ # ./runme.sh cython_benchmark 0 5

for (( i=$2; i<=$3; i++ ))
do
	time python "$1.py" $i
done