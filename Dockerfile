# Use NVIDIA CUDA base image with Ubuntu 22.04 and development tools
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and pip
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to use python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a symlink from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Verify that python and nvcc are available
RUN python --version && nvcc --version

# Install elan (Lean version manager) and set up Lean 4
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | \
    sh -s -- --default-toolchain leanprover/lean4:stable --no-modify-path -y

# Set up environment variables for elan
ENV PATH="/root/.elan/bin:${PATH}"

# Set the working directory
WORKDIR /root

# Copy requirements.txt
COPY requirements.txt /root/

# Copy mathlib4
COPY mathlib4 /root/mathlib4

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install flash-attn
RUN pip3 install -U flash-attn --no-build-isolation

# Build mathlib4
WORKDIR /root/mathlib4
# [below] doesn't do anything because docker build is run in an isloated environment
# RUN lake exe cache get
RUN lake build