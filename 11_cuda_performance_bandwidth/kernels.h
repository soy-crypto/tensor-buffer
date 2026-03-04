#pragma once

__global__
void coalesced_read(float* input, float* output, int N);

__global__
void strided_read(float* input, float* output, int N, int stride);

__global__
void occupancy_test(float* input, float* output);