# Sypha - Interior Point Optimization on GPU with CUDA
# Uses NVIDIA CUDA development image for Linux build (current, non-deprecated tag)
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies (GSL, Boost, make)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgsl-dev \
    libboost-all-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sypha

# Copy project files
COPY Makefile .
COPY src/ src/
COPY include/ include/
COPY data/ data/
COPY examples/ examples/

# Create bin directory for object files
RUN mkdir -p bin

# Build main binary and example (Makefile uses BOOST=/usr and INCLUDES=-Isrc -Iinclude by default)
RUN make && make scp_solver

# Default: run the demo via the new API
CMD ["./scp_solver", "data/demo00.txt", "100"]
