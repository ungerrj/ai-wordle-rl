"""
Custom Wordle environment for reinforcement learning using Gymnasium.
This environment allows an agent to play Wordle by making guesses and receiving feedback.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from .word_lists import DEFAULT_ACCEPTED, DEFAULT_SOLUTIONS, load_word_list

class WordleEnv(gym.Env):
    """
    Custom Gymnasium environment for Wordle game.

    Observation:
        - Current guess (5 letter word)
        - Feedback from previous guesses (5-tuple of colors)
        - Number of remaining guesses

    Action:
        - An index into the accepted word list (every accepted word is a legal guess)

    Target:
        - Drawn from the solution list, which must be a subset of the accepted list

    Reward:
        - +100 for correct guess
        - -1 for each guess used
        - -0.1 for invalid guess
    """

    def __init__(self, accepted_list: str = DEFAULT_ACCEPTED, solution_list: str = DEFAULT_SOLUTIONS,
                 max_attempts: int = 6):
        super(WordleEnv, self).__init__()

        # Load word lists by name (see word_lists.WORD_LISTS)
        self.accepted_list = accepted_list
        self.solution_list = solution_list
        self.accepted_words = load_word_list(accepted_list)
        self.solution_words = load_word_list(solution_list)
        not_accepted = sorted(set(self.solution_words) - set(self.accepted_words))
        if not_accepted:
            raise ValueError(
                f"Solution list '{solution_list}' has {len(not_accepted)} words missing from "
                f"accepted list '{accepted_list}', e.g. {not_accepted[:5]}"
            )
        self.max_attempts = max_attempts
        self.target_word = ""
        self.attempts = 0
        self.guesses = []
        self.feedback = []

        # Define action and observation spaces
        # Action space: one action per accepted word
        self.action_space = spaces.Discrete(len(self.accepted_words))

        # Observation space:
        # - Current guess (5 letters)
        # - Previous feedback (5x3 for each letter: green=2, yellow=1, gray=0)
        # - Remaining attempts
        self.observation_space = spaces.Dict({
            "guess": spaces.MultiDiscrete([26] * 5),  # 5 letters
            "feedback": spaces.MultiDiscrete([3] * 5),  # 5 feedback values
            "remaining_attempts": spaces.Discrete(max_attempts + 1)
        })

        # Initialize game state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Select a random target word from the solutions
        self.target_word = random.choice(self.solution_words)
        self.attempts = 0
        self.guesses = []
        self.feedback = []

        # Return initial observation
        return self._get_observation(), {}

    def _get_observation(self) -> Dict:
        """Get current observation."""
        # Initialize with zeros
        guess = [0] * 5  # Placeholder for letters
        feedback = [0] * 5  # Placeholder for feedback

        if len(self.guesses) > 0:
            # Use the most recent guess
            latest_guess = self.guesses[-1]
            guess = [ord(c) - ord('a') for c in latest_guess]

            # Use the most recent feedback
            feedback = self.feedback[-1]

        return {
            "guess": np.array(guess, dtype=np.int64),
            "feedback": np.array(feedback, dtype=np.int64),
            "remaining_attempts": self.max_attempts - self.attempts
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        # Get the guessed word
        if action >= len(self.accepted_words):
            # Invalid action
            reward = -0.1
            done = False
            info = {"invalid_action": True}
            return self._get_observation(), reward, done, False, info

        guess_word = self.accepted_words[action]

        # Validate word is 5 letters
        if len(guess_word) != 5:
            reward = -0.1
            done = False
            info = {"invalid_action": True, "reason": "Word must be 5 letters"}
            return self._get_observation(), reward, done, False, info

        # Record the guess
        self.attempts += 1
        self.guesses.append(guess_word)

        # Calculate feedback
        feedback = self._calculate_feedback(guess_word)
        self.feedback.append(feedback)

        # Calculate reward
        reward = self._calculate_reward(guess_word, feedback)

        # Check if game is done
        done = self._is_done(guess_word, feedback)

        # Check if max attempts reached
        if self.attempts >= self.max_attempts:
            done = True

        # Return observation, reward, done, info
        info = {"guess": guess_word, "feedback": feedback, "target": self.target_word}
        return self._get_observation(), reward, done, False, info

    def _calculate_feedback(self, guess: str) -> List[int]:
        """Calculate feedback for a guess.

        Feedback codes:
        - 0: Gray (letter not in word)
        - 1: Yellow (letter in word but wrong position)
        - 2: Green (letter in correct position)
        """
        feedback = [0] * 5
        target_chars = list(self.target_word)
        guess_chars = list(guess)

        # First pass: mark greens
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback[i] = 2
                target_chars[i] = None  # Mark as used

        # Second pass: mark yellows and grays
        for i in range(5):
            if feedback[i] == 2:
                continue  # Already marked as green

            if guess_chars[i] in target_chars:
                feedback[i] = 1
                # Find the first unused occurrence
                idx = target_chars.index(guess_chars[i])
                target_chars[idx] = None  # Mark as used
            else:
                feedback[i] = 0

        return feedback

    def _calculate_reward(self, guess: str, feedback: List[int]) -> float:
        """Calculate reward for a guess."""
        # Check if guess is correct
        if guess == self.target_word:
            return 100.0

        # Penalty for each guess
        reward = -1.0

        # Additional penalty for invalid guesses
        if len(guess) != 5:
            reward -= 0.1

        return reward

    def _is_done(self, guess: str, feedback: List[int]) -> bool:
        """Check if the game is done."""
        return guess == self.target_word or self.attempts >= self.max_attempts

    def render(self, mode='human'):
        """Render the environment state."""
        print(f"Target word: {self.target_word}")
        print(f"Remaining attempts: {self.max_attempts - self.attempts}")
        print("Guesses:")
        for i, (guess, fb) in enumerate(zip(self.guesses, self.feedback)):
            print(f"  {i+1}: {guess} - {fb}")