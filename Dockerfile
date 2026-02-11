# Sypha - Interior Point Optimization on GPU with CUDA
# Uses NVIDIA CUDA development image for Linux build
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

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
COPY data/ data/

# Create bin directory for object files
RUN mkdir -p bin

# Build (Makefile uses BOOST=/usr and INCLUDES=-Isrc by default)
RUN make

# Default: run the demo
CMD ["./sypha", "--verbosity", "100", "--model", "SCP", "--input-file", "data/demo00.txt"]
