# AI Wordle Solver with Reinforcement Learning

This project implements a reinforcement learning agent to solve Wordle puzzles using Gymnasium.

## Project Structure

- `environment/` - Custom Wordle environment following Gymnasium interface
- `agents/` - Reinforcement learning agents (DQN implementation)
- `training/` - Training scripts and configurations
- `utils/` - Utility functions
- `notebooks/` - Jupyter notebooks for experimentation
- `results/` - Training results, one folder per word list variant (`results/<accepted>/<solutions>/`)

## Requirements

- Docker 20.10+
- ROCm 10.0 compatible AMD GPU

## Setup

The project runs in Docker on top of `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`,
which already includes Python 3.14 and PyTorch 2.13.0 with ROCm 10.0. The Dockerfile installs the
remaining packages from `requirements.txt`, using `rocm-constraints.txt` so pip keeps the image's
ROCm build of PyTorch.

```bash
# Build the image (rebuild after code changes; the code is copied in at build time)
docker compose build

# Open a shell in the container with GPU access
docker compose run --rm wordle

# Or run a single command
docker compose run --rm wordle python -m training.train --episodes 1000
```

`compose.yaml` passes the GPU devices through and maps `results/` and `data/` to the host, so
training output and the downloaded word lists land in your working tree. The container runs as your
user (UID/GID 1000 by default) so those files are yours. If your IDs differ, set `HOST_UID`,
`HOST_GID` and `RENDER_GID` (the host's `render` group, from `getent group render`) in the
environment or a `.env` file.

## Usage

Inside the container (working directory `/app`):

```bash
# Train the agent (downloads the word lists to data/word_lists/ on first run)
python -m training.train --episodes 1000

# Choose the word lists: --accepted is what the agent may guess, --solutions is what targets are drawn from
python -m training.train --accepted tabatkins-14855 --solutions cfreshman-2315

# Run the demo
python example_usage.py
```

## Testing

```bash
# Inside the container:
python test_setup.py
python test_core_components.py

# Or from the host:
docker compose run --rm wordle python test_setup.py
```

## Continuous Integration

`.github/workflows/ci.yml` runs on every push (and manually from the Actions tab). It builds the
image with Compose and then, inside the container on the GPU:
- lints for syntax errors and undefined names
- runs `test_core_components.py`, `test_setup.py` and `example_usage.py`
- re-downloads every pinned word list and checks its size
- does a short training run and checks the checkpoints and plot were written

The image is about 52 GB, too big for GitHub's hosted runners, so CI runs on a self-hosted runner on
the GPU machine. Jobs only run while that machine is on and the runner service is running.

### Registering the runner (one time)

1. On GitHub, open the repo's **Settings → Actions → Runners → New self-hosted runner** and choose
   Linux x64. Run the download commands it shows (e.g. into `~/actions-runner`).
2. Configure it with the `rocm` label, which the workflow requires:
   ```bash
   ./config.sh --url https://github.com/ungerrj/ai-wordle-rl --token <TOKEN> --labels rocm
   ```
3. Install it as a service running as your user. That user needs to be in the `docker`, `video` and
   `render` groups:
   ```bash
   sudo ./svc.sh install $USER && sudo ./svc.sh start
   ```
4. In **Settings → Actions → General**, under fork pull request workflows, require approval for all
   external contributors.

The repo is public, and a self-hosted runner executes workflow code on your machine with Docker
access. That's why the workflow triggers only on `push` and manual runs. Don't add `pull_request` or
`pull_request_target` triggers. The step 4 setting is a backstop in case one is added.

CI builds under the tag `ai-wordle-rl:ci` so it doesn't replace your local `ai-wordle-rl` image.

## Features

- Custom Gymnasium environment for Wordle
- Deep Q-Network (DQN) reinforcement learning agent
- Training script with configurable parameters
- Visualization of training progress
- Modular design for easy extension

See `troubleshooting.md` for build and GPU access issues.
