# Base image with NVIDIA drivers and OpenGL support
FROM nvidia/cudagl:11.4.2-base-ubuntu20.04

# Set non-interactive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - build-essential: for compiling C++ code
# - python3-dev: for python C extensions
# - x11-apps: for GUI forwarding
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.8 \
    python3.8-dev \
    python3-pip \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Make python3.8 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Set up the working directory
WORKDIR /app

# Copy and install Python dependencies first to leverage Docker caching
COPY ../requirements.txt .
RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Install the custom dvrk_gym package in editable mode
RUN pip3 install -e .

# Set a default command to keep the container running
CMD ["/bin/bash"]
