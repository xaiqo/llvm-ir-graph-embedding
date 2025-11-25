FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 17 all && \
    rm llvm.sh

RUN ln -sf /usr/bin/clang-17 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-17 /usr/bin/clang++ && \
    ln -sf /usr/bin/llvm-config-17 /usr/bin/llvm-config && \
    ln -sf /usr/bin/opt-17 /usr/bin/opt && \
    ln -sf /usr/bin/llc-17 /usr/bin/llc


COPY ml_core/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# 3. Environment Setup
WORKDIR /workspace
ENV LLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
ENV PYTHONPATH=/workspace

RUN llvm-config --version && clang --version && python3 --version

CMD ["/bin/bash"]

