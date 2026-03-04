#!/bin/bash
mkdir -p build
cd build
cmake ..
make
./inference_runtime
