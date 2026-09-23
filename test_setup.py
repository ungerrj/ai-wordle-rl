#!/usr/bin/env python3
"""
Quick test to verify the project setup is working correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import gymnasium
        print("✓ Gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Gymnasium: {e}")
        return False

    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False

    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False

    try:
        from environment.wordle_env import WordleEnv
        print("✓ WordleEnv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WordleEnv: {e}")
        return False

    try:
        from agents.dqn_agent import DQNAgent
        print("✓ DQNAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DQNAgent: {e}")
        return False

    return True

def test_environment_registration():
    """Test that the custom environment is registered."""
    print("\nTesting environment registration...")

    try:
        import gymnasium as gym
        # Check if our environment is registered
        env_ids = [spec.id for spec in gym.envs.registry.values() if 'Wordle' in spec.id]
        print(f"✓ Registered environments: {env_ids}")

        if 'WordleEnv-v0' in env_ids:
            print("✓ WordleEnv-v0 is properly registered")
            return True
        else:
            print("✗ WordleEnv-v0 not found in registry")
            return False

    except Exception as e:
        print(f"✗ Error testing environment registration: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        import gymnasium as gym
        from environment.wordle_env import WordleEnv
        from agents.dqn_agent import DQNAgent

        # Test environment creation
        env = gym.make('WordleEnv-v0', accepted_list='sample-20', solution_list='sample-20')
        print("✓ Environment created successfully")

        # Test agent creation
        agent = DQNAgent(action_space=env.action_space.n)
        print("✓ Agent created successfully")

        # Test reset
        state, _ = env.reset()
        print("✓ Environment reset successful")

        # Test a simple step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")

        print("✓ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing Wordle RL Project Setup ===\n")

    tests = [
        test_imports,
        test_environment_registration,
        test_basic_functionality
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Test Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("✓ All tests passed! The project is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())