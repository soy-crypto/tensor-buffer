#!/bin/bash

set -e

echo "================================="
echo "Checking GPU Foundations Projects"
echo "================================="

echo ""
echo "1. Checking C++ inference runtime"
echo "---------------------------------"

cd cpp_inference_runtime

mkdir -p build
cd build

cmake .. > /dev/null
make

echo "Running inference runtime..."
./inference_runtime || echo "Runtime error in inference runtime"

cd ../..

echo ""
echo "2. Checking CUDA kernel optimization"
echo "-------------------------------------"

cd cuda_kernel_optimization

echo "Compiling vector_add..."
nvcc vector_add/main.cu -o vector_add.out
./vector_add.out || echo "Runtime error in vector_add"

echo "Compiling naive GEMM..."
nvcc gemm_naive/main.cu -o gemm_naive.out
./gemm_naive.out || echo "Runtime error in gemm_naive"

echo "Compiling tiled GEMM..."
nvcc gemm_tiled/main.cu -o gemm_tiled.out
./gemm_tiled.out || echo "Runtime error in gemm_tiled"

cd ..

echo ""
echo "3. Checking CUDA microbenchmarks"
echo "---------------------------------"

cd cuda_microbenchmarks/bandwidth_test

make
./run.sh || echo "Error running bandwidth benchmark"

cd ../..

echo ""
echo "4. Checking LLM inference benchmarks"
echo "-------------------------------------"

cd llm_inference_systems/inference_benchmarks

python benchmark.py || echo "Python benchmark failed"

cd ../../..

echo ""
echo "================================="
echo "All checks completed."
echo "================================="

