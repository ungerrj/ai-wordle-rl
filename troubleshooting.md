# Troubleshooting Docker-Based Installation

## Common Docker Build Issues and Solutions

### 1. Package Build Failure (pkg_resources error)

**Error**: `ModuleNotFoundError: No module named 'pkg_resources'` while pip builds a package from source

**Cause**: The package version has no Python 3.14 wheel, so pip tried a source build, and those builds depend on `pkg_resources`, which Python 3.14 environments no longer have.

**Solution**: The Dockerfile installs with `--only-binary=:all:`, so this should surface as "no matching distribution" instead. Raise the version floor in `requirements.txt` to a release that ships Python 3.14 wheels.

### 2. Docker Image Build Failures

**Error**: Build fails during `apt-get update` or package installation

**Solutions**:
- Check internet connectivity
- Try: `docker build --no-cache -t ai-wordle-rl .` to force fresh download
- Verify Docker daemon is running properly
- Increase Docker's resource allocation (memory, disk) if builds hang

**Error**: Base image not found: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`

**Solutions**:
- Verify the image name is correct
- Try pulling first: `docker pull rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`
- If using a different ROCm/PyTorch version, adjust the FROM line in Dockerfile
- Check if the image exists on Docker Hub: `docker search rocm/pytorch`

### 3. GPU Access Issues in Container

**Error**: "No ROCm support found" or HIP not available inside container

**Solutions**:
- Always run with GPU access flags:
  ```bash
  docker run --device=/dev/kfd --device=/dev/dri --group-add video -it ai-wordle-rl bash
  ```
- Verify host ROCm installation: `/opt/rocm/bin/rocminfo` (should work on host)
- Check user is in video group: `groups $USER` (should include "video")
- If not: `sudo usermod -aG video $USER` then log out and back in

**Error**: Permission denied accessing /dev/kfd or /dev/dri

**Solutions**:
- The `--group-add video` flag is essential for GPU access
- Ensure host user is in video group
- Log out and back in after changing group membership
- As last resort (less secure): add `--privileged` flag for testing only

### 4. PyTorch ROCm Verification

**Problem**: Unsure if PyTorch is properly using ROCm inside container

**Solution**: Run the GPU check in [Verification Steps](#verification-steps) step 1. ROCm builds of PyTorch use the `torch.cuda` API, so a working GPU shows `GPU available: True` even though no CUDA is involved.

If `GPU available` is `False` but `HIP version` shows a version, the image's PyTorch is fine and the container can't see the GPU: it was started without `--device=/dev/kfd --device=/dev/dri --group-add video` (see section 3).

### 5. General Testing Failures

**Error**: Tests pass but training fails or behaves unexpectedly

**Solutions**:
- Run the full test suite: `python test_setup.py`
- Try example usage: `python example_usage.py`
- Check results/ directory for logs and output
- Verify numpy version compatibility with PyTorch ROCm build

## Verification Steps

After successful build and run:

1. **Check PyTorch can use the GPU**:
   ```bash
   docker run --device=/dev/kfd --device=/dev/dri --group-add video --rm ai-wordle-rl \
   python -c "import torch; print('HIP version:', torch.version.hip); print('CUDA version:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
   ```
   A working setup shows:
   - `HIP version`: any version (a ROCm build of PyTorch)
   - `CUDA version`: `None` (not a CUDA build)
   - `GPU available`: `True`
   - `GPU`: your AMD card's name

2. **Verify basic functionality**:
   ```bash
   docker run --device=/dev/kfd --device=/dev/dri --group-add video --rm ai-wordle-rl \
   python test_setup.py
   ```

3. **Test example execution**:
   ```bash
   docker run --device=/dev/kfd --device=/dev/dri --group-add video --rm ai-wordle-rl \
   python example_usage.py
   ```

## Notes

- The Dockerfile is optimized to work with your specified base image which already includes PyTorch 2.13.0+rocm10.0
- We avoid reinstalling PyTorch to prevent version conflicts with the ROCm build
- All other packages (gymnasium, numpy, matplotlib, pandas, seaborn) are installed via pip
- If you continue to experience issues, check:
  1. Docker version (20.10+ recommended)
  2. Host ROCm installation and permissions
  3. Internet connectivity for Docker build
  4. Available system resources (RAM, disk space)