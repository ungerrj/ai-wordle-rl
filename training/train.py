"""
Training script for Wordle reinforcement learning agent.
This script trains the DQN agent on the Wordle environment.
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.wordle_env import WordleEnv
from agents.dqn_agent import DQNAgent
import os
import argparse

def train_agent(episodes: int = 1000,
                save_interval: int = 100,
                render: bool = False):
    """
    Train the DQN agent on the Wordle environment.

    Args:
        episodes: Number of training episodes
        save_interval: How often to save the model
        render: Whether to render the environment during training
    """

    # Create environment
    env = gym.make('WordleEnv-v0')

    # Create agent
    agent = DQNAgent(action_space=len(env.unwrapped.word_list))

    # Training statistics
    scores = []
    episodes_list = []

    print(f"Starting training for {episodes} episodes...")

    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()

        total_reward = 0
        done = False

        # Play one episode
        while not done:
            # Choose action
            action = agent.act(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Check if done
            done = terminated or truncated

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state

            # Accumulate reward
            total_reward += reward

            # Train the agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        # Record score
        scores.append(total_reward)
        episodes_list.append(episode)

        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f"Episode {episode}, Average Score (last 10): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            model_path = f"results/model_episode_{episode}.pth"
            agent.save_model(model_path)
            print(f"Model saved to {model_path}")

    # Save final model
    agent.save_model("results/final_model.pth")
    print("Training completed!")

    # Plot training progress
    plot_training_progress(episodes_list, scores)

    return agent, scores

def plot_training_progress(episodes: list, scores: list):
    """Plot training progress."""
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores, label='Episode Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_progress.png')
    plt.close()
    print("Training progress plot saved to results/training_progress.png")

def main():
    """Main training function."""

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Wordle RL agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save_interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')

    args = parser.parse_args()

    # Train the agent
    agent, scores = train_agent(
        episodes=args.episodes,
        save_interval=args.save_interval,
        render=args.render
    )

    print("Training finished!")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")

if __name__ == "__main__":
    main()