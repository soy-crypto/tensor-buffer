#include <iostream>
#include "kernels.h"

int main() {

    int N = 1 << 26;

    float *d_input, *d_output;

    cudaMalloc(&d_input, N*sizeof(float));
    cudaMalloc(&d_output, N*sizeof(float));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    coalesced_read<<<grid, block>>>(d_input, d_output, N);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}