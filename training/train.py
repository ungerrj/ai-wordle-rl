"""
Training script for Wordle reinforcement learning agent.
This script trains the DQN agent on the Wordle environment.
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.wordle_env import WordleEnv
from environment.word_lists import DEFAULT_ACCEPTED, DEFAULT_SOLUTIONS, WORD_LISTS, load_word_list
from agents.dqn_agent import DQNAgent
import os
import argparse

def train_agent(env: gym.Env,
                episodes: int = 1000,
                save_interval: int = 100,
                render: bool = False):
    """
    Train the DQN agent on the Wordle environment.

    Args:
        env: The Wordle environment to train on
        episodes: Number of training episodes
        save_interval: How often to save the model
        render: Whether to render the environment during training
    """

    # Create agent
    agent = DQNAgent(action_space=len(env.unwrapped.accepted_words))

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

def confirm_device() -> bool:
    """Report the training device, asking before falling back to CPU.

    Returns True to continue training, False to abort. Without an interactive
    terminal there is no one to ask, so a missing GPU aborts.
    """
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        return True

    print("No GPU detected. If you're in Docker, check that the container was started with")
    print("  --device=/dev/kfd --device=/dev/dri --group-add video")
    try:
        answer = input("Train on CPU instead? This is much slower. [y/N] ")
    except EOFError:
        print("\nNo terminal to answer from; aborting.")
        return False
    return answer.strip().lower() in ("y", "yes")

def main():
    """Main training function."""

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Wordle RL agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save_interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    parser.add_argument('--accepted', default=DEFAULT_ACCEPTED, choices=WORD_LISTS,
                        help='Word list the agent may guess from (its action space)')
    parser.add_argument('--solutions', default=DEFAULT_SOLUTIONS, choices=WORD_LISTS,
                        help='Word list target words are drawn from')
    parser.add_argument('--refresh_word_lists', action='store_true',
                        help='Re-download the word lists even if they are cached')

    args = parser.parse_args()

    if args.refresh_word_lists:
        for name in (args.accepted, args.solutions):
            load_word_list(name, refresh=True)

    # Create the environment first so a word list problem fails before the device prompt
    env = gym.make('WordleEnv-v0', accepted_list=args.accepted, solution_list=args.solutions)
    print(f"Accepted guesses: {args.accepted} ({len(env.unwrapped.accepted_words)} words), "
          f"solutions: {args.solutions} ({len(env.unwrapped.solution_words)} words)")

    if not confirm_device():
        raise SystemExit("Training aborted.")

    # Train the agent
    agent, scores = train_agent(
        env,
        episodes=args.episodes,
        save_interval=args.save_interval,
        render=args.render
    )

    print("Training finished!")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Average score over last 100 episodes: {np.mean(scores[-100:]):.2f}")

if __name__ == "__main__":
    main()