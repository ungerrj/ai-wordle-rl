# Dockerfile for AI Wordle RL Project with ROCm Support
# Using the exact base image recommended by the user:
# rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
# Note: This base image already includes PyTorch 2.13.0 with ROCm 10.0,
# so we avoid reinstalling PyTorch to prevent version conflicts.

# Use the ROCm PyTorch base image exactly as specified
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/robot

# compose.yaml runs as the host user, which needs a writable HOME for
# caches (matplotlib, MIOpen)
RUN mkdir -p $HOME && chmod 1777 $HOME

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies from requirements.txt. rocm-constraints.txt pins
# torch to the base image's ROCm build so pip keeps it instead of pulling a
# PyPI wheel. --only-binary avoids source builds, which need the removed
# pkg_resources module on Python 3.14.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --only-binary=:all: -r requirements.txt -c rocm-constraints.txt && \
    # Install additional development tools
    pip install jupyterlab pytest black flake8

# Expose ports for Jupyter if needed
EXPOSE 8888

# Default command
CMD ["bash"]