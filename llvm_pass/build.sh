#!/bin/bash
set -e

mkdir -p build
cd build

cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm

ninja

echo "Build complete. Plugin is at $(pwd)/GraphExtractor.so"

