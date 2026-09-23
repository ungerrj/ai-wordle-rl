"""
Helper functions for the Wordle RL project.
"""

import numpy as np
import torch
from typing import List, Tuple
import matplotlib.pyplot as plt

def plot_training_history(scores: List[float], window_size: int = 10):
    """
    Plot training scores with moving average.

    Args:
        scores: List of episode scores
        window_size: Size of moving average window
    """
    if len(scores) == 0:
        return

    # Calculate moving average
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        x_vals = range(window_size-1, len(scores))
    else:
        moving_avg = scores
        x_vals = range(len(scores))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, alpha=0.5, label='Episode Score')
    plt.plot(x_vals, moving_avg, linewidth=2, label=f'{window_size}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_history.png')
    plt.close()

def get_device():
    """Get the appropriate device for training (CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_word(word: str, word_list: List[str]) -> bool:
    """
    Validate if a word is in the valid word list.

    Args:
        word: Word to validate
        word_list: List of valid words

    Returns:
        True if word is valid, False otherwise
    """
    return word.lower() in [w.lower() for w in word_list]

def format_feedback(feedback: List[int]) -> str:
    """
    Format feedback for display.

    Args:
        feedback: List of feedback codes (0=gray, 1=yellow, 2=green)

    Returns:
        Formatted string representation
    """
    symbols = {0: '🔲', 1: '🟨', 2: '🟩'}
    return ''.join(symbols[code] for code in feedback)