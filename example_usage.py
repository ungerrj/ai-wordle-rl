"""
Example usage of the Wordle RL environment and agent.
This script demonstrates how to use the implemented components.
"""

import gymnasium as gym
from environment.wordle_env import WordleEnv
from agents.dqn_agent import DQNAgent
import torch

def test_environment():
    """Test the custom Wordle environment."""
    print("Testing Wordle Environment...")

    # Create environment
    env = gym.make('WordleEnv-v0', accepted_list='sample-20', solution_list='sample-20')

    # Reset environment
    state, _ = env.reset()
    print(f"Initial state: {state}")
    print(f"Action space size: {env.action_space.n}")

    # Test a few steps
    for i in range(3):
        # Get a random action
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Done: {terminated or truncated}")
        print(f"  Info: {info}")

        if terminated or truncated:
            break

def test_agent():
    """Test the DQN agent."""
    print("\nTesting DQN Agent...")

    # Create environment
    env = gym.make('WordleEnv-v0', accepted_list='sample-20', solution_list='sample-20')

    # Create agent
    agent = DQNAgent(action_space=env.action_space.n)

    # Test agent actions
    state, _ = env.reset()

    # Get a few actions from the agent
    for i in range(3):
        action = agent.act(state, training=True)
        print(f"Agent action: {action}")

        # Try to get a valid word from the action
        try:
            word = env.unwrapped.accepted_words[action]
            print(f"  Selected word: {word}")
        except IndexError:
            print("  Invalid action (index out of bounds)")

        # Do a step to see what happens
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        if terminated or truncated:
            break

def main():
    """Main function to demonstrate usage."""
    print("=== Wordle RL Project Demo ===")

    try:
        test_environment()
        test_agent()
        print("\n=== Demo completed successfully ===")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()