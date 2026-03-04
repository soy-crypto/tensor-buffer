#!/bin/bash

set -e

echo "Finalizing GPU Foundations repository..."

echo "Creating build folders..."

mkdir -p cpp_inference_runtime/build
mkdir -p cuda_kernel_optimization/build
mkdir -p cuda_microbenchmarks/build

echo "Creating example run scripts..."

# C++ inference runtime run script
cat << 'EOF' > cpp_inference_runtime/run.sh
#!/bin/bash
mkdir -p build
cd build
cmake ..
make
./inference_runtime
EOF

chmod +x cpp_inference_runtime/run.sh


# CUDA kernel optimization script
cat << 'EOF' > cuda_kernel_optimization/run.sh
#!/bin/bash

echo "Compiling CUDA kernels..."

nvcc vector_add/main.cu -o vector_add.out
nvcc gemm_naive/main.cu -o gemm_naive.out
nvcc gemm_tiled/main.cu -o gemm_tiled.out

echo "Running kernels..."

./vector_add.out
./gemm_naive.out
./gemm_tiled.out
EOF

chmod +x cuda_kernel_optimization/run.sh


# CUDA microbench run script
cat << 'EOF' > cuda_microbenchmarks/run.sh
#!/bin/bash

cd bandwidth_test
make
./run.sh
EOF

chmod +x cuda_microbenchmarks/run.sh


echo "Creating placeholder diagrams..."

touch diagrams/inference_runtime_architecture.png
touch diagrams/cuda_memory_hierarchy.png
touch diagrams/llm_inference_stack.png


echo "Creating .gitignore..."

cat << 'EOF' > .gitignore
build/
*.out
*.o
*.ptx
*.engine
*.log
__pycache__/
*.pyc
EOF


echo ""
echo "Final repo structure:"
echo ""

ls -R

echo ""
echo "GPU systems repository ready for GitHub."
