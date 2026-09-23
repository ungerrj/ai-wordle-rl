"""
Constants and configuration values for the Wordle RL project.
"""

# Environment constants
MAX_ATTEMPTS = 6
WORD_LENGTH = 5

# Training constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 1000

# Model constants
HIDDEN_SIZE = 256

# File paths
WORD_LIST_FILE = "words.txt"
RESULTS_DIR = "results"
MODEL_SAVE_PATH = "results/final_model.pth"