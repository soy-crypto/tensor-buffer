#!/bin/bash

echo "Compiling CUDA kernels..."

nvcc vector_add/main.cu -o vector_add.out
nvcc gemm_naive/main.cu -o gemm_naive.out
nvcc gemm_tiled/main.cu -o gemm_tiled.out

echo "Running kernels..."

./vector_add.out
./gemm_naive.out
./gemm_tiled.out
