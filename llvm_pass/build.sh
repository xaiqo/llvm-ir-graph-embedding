#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm

make -j$(nproc)

echo "Build complete. Plugin is at $(pwd)/lib/GraphExtractor.so"
