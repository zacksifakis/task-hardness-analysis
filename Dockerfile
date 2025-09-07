# Use a known-stable base image
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# # Set non-interactive to prevent prompts during build
# ENV DEBIAN_FRONTEND=noninteractive

# # Layer 1: Install system dependencies.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10 \
#     python3.pip && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Set the working directory
# WORKDIR /workspace

# # Layer 2: Install vLLM and its MISSING dependency, 'outlines'.
# # This is the direct fix for the 'No module named outlines.fsm' error.
# # We also pin numpy here to prevent all binary conflicts.
# RUN python3 -m pip install --upgrade pip && \
#     python3 -m pip install --no-cache-dir \
#     "numpy~=1.26.4" \
#     "vllm==0.3.3" \
#     "outlines==0.0.34"

# # Layer 3: Install your application's other packages.
# COPY requirements.txt .
# RUN python3 -m pip install --no-cache-dir -r requirements.txt

# # Layer 4: Copy the rest of your project code
# COPY . .

#!/bin/bash

#!/bin/bash

#!/bin/bash

# Use a known-stable base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive to prevent prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Set the environment variable to disable Numba's broken cache.
# This is the direct fix for the "no locator available" error.
ENV NUMBA_CACHE_DIR=/tmp

# Set the working directory
WORKDIR /workspace

# Layer 3: Install build tools and PIN NUMPY to the version
# that the vLLM pre-built wheel was compiled against. This is critical.
RUN python3 -m pip install --upgrade pip "numpy~=1.26.4"

# Layer 4: Install vLLM and its MISSING dependency, 'outlines'.
# This will pull in the correct, compatible versions of torch and flash-attn.
RUN python3 -m pip install --no-cache-dir "vllm==0.3.3" "outlines==0.0.34"

# Layer 5: Install your application's other packages.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Layer 6: Copy the rest of your project code
COPY . .