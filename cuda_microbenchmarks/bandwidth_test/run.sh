#!/bin/bash

make clean
make

./benchmark
ncu ./benchmark