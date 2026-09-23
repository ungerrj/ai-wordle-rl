# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This is a personal project with a single target: Ubuntu 26 host, AMD GPU (ROCm), running in Docker. Don't add alternate install paths (local venvs, CPU-only or CUDA setups, extra requirements files, multi-version Python support).

- Base image `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` provides Python 3.14 and PyTorch 2.13.0+rocm.
- `requirements.txt` is the only dependency list. The Dockerfile installs it with `-c rocm-constraints.txt` so pip keeps the image's ROCm torch, and with `--only-binary=:all:` because source builds fail on 3.14 (no `pkg_resources`). New dependencies need versions that ship cp314 wheels.
- ROCm PyTorch uses the `torch.cuda` API: `torch.cuda.is_available()` is `True` on a working AMD GPU, and devices are `cuda:0`.

## Commands

```bash
docker build -t ai-wordle-rl .
docker run --device=/dev/kfd --device=/dev/dri --group-add video -it --rm ai-wordle-rl bash
```

All three device flags are required for GPU access. Without them, `torch.cuda.is_available()` is `False` and training asks before falling back to CPU (it aborts with exit 1 if there's no terminal to answer from, e.g. `docker run` without `-i`).

The Dockerfile copies the code in at build time, so either rebuild after edits or mount the working tree: add `-v "$PWD":/app` to test current code without rebuilding.

Inside the container (all commands run from the repo root, `/app`; the packages are imported as top-level modules):

```bash
python -m training.train --episodes 1000 --save_interval 100   # not `python training/train.py` (ModuleNotFoundError)
python example_usage.py                                        # env + agent smoke demo
python test_setup.py                                           # imports, env registration, reset/step
python test_core_components.py                                 # imports only
python -c "import test_setup as t; t.test_basic_functionality()"   # run a single check
```

The test files are plain scripts: each check returns `True`/`False` and `main()` sets the exit code. `pytest` collects them but reports failed checks as passing (it only warns about the return value), so run them as scripts. Adding `-W error::UserWarning` surfaces gymnasium's observation-space checker warnings as failures.

`black` and `flake8` are installed in the image but the code has never been formatted or linted with them (flake8 reports ~90 issues), so don't reformat whole files as part of unrelated changes.

## Architecture

- **`environment/`**: `WordleEnv` (a `gymnasium.Env`). Importing the `environment` package registers `WordleEnv-v0` with `max_episode_steps=100`, so `gym.make('WordleEnv-v0')` only works after `environment` has been imported, and it returns the env wrapped in `TimeLimit`. Gymnasium 1.x wrappers don't forward attributes, so use `env.unwrapped.accepted_words` etc.
  - The env takes two named word lists from `environment/word_lists.py`: `accepted_list` (legal guesses; default `tabatkins-14855`) and `solution_list` (possible targets; default `cfreshman-2315`). Actions are indexes into `accepted_words`, so the action space is `Discrete(len(accepted_words))` and every action is a legal guess. `reset()` draws the target only from `solution_words`, which must be a subset of the accepted list (the env raises otherwise). The agent is never shown the solution list.
  - Remote lists are pinned to an upstream revision and checked against the size in their name, so a name always means the same words in the same order (checkpoints depend on that order). They're downloaded on first use to `data/word_lists/` (gitignored, next to `environment/`), and a failed download raises; there's no fallback list. To add a variant, add an entry to `WORD_LISTS` named `<source>-<count>`.
  - Tests and demos use the inline `sample-20` list for both (`gym.make('WordleEnv-v0', accepted_list='sample-20', solution_list='sample-20')`) so they don't need the network.
  - Observations are `{"guess": int64[5] letter codes, "feedback": int64[5] (0 gray/1 yellow/2 green), "remaining_attempts": int}`, declared as `MultiDiscrete`/`Discrete`. Keep returned values matching the declared space or gymnasium's env checker warns on every reset/step.
  - Rewards: +100 for solving, -1 per guess. An out-of-range action gives -0.1 and does not use up an attempt.
- **`agents/dqn_agent.py`**: `DQN` network and `DQNAgent` (epsilon-greedy, replay buffer of raw observation dicts, target network). `_state_to_tensor` one-hot encodes an observation into the 136-dim network input (5×26 letters + 5×3 feedback + normalized remaining attempts). That encoding, `DQN`'s default `input_size`, and the env's observation shape must change together.
  - `DQN`'s output size is the agent's `action_space` (one Q-value per word), so a checkpoint only loads into an agent built for the same accepted list. Checkpoints don't record which lists they were trained with.
  - `DQN` has no dropout or batch norm, so its outputs don't depend on `train()`/`eval()` mode. Adding either means switching modes in `act()` and keeping the target network in `eval()`.
  - Epsilon decays once per `replay()` call (per training step, not per episode).
- **`training/train.py`**: training loop. `main()` creates `results/`, builds the env from `--accepted`/`--solutions` (so a word list problem fails before any prompt; `--refresh_word_lists` re-downloads), runs `confirm_device()` (GPU check / CPU prompt), trains, and writes `results/model_episode_N.pth`, `results/final_model.pth`, and `results/training_progress.png`. `results/` is gitignored except `.gitkeep`.
- **`utils/`**: `constants.py` and `helpers.py` duplicate hyperparameters and helpers, but nothing imports them; the values actually used are the defaults in `DQNAgent.__init__` and `WordleEnv.__init__`.
