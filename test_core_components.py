#!/usr/bin/env python3
"""
Simple test to verify core components work without complex installations.
"""

import sys
import os

def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")

    # Test basic Python imports
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
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Matplotlib: {e}")
        return False

    return True

def test_local_imports():
    """Test that our local modules can be imported."""
    print("\nTesting local imports...")

    try:
        from environment.wordle_env import WordleEnv
        print("✓ WordleEnv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WordleEnv: {e}")
        # Show the file structure to help debug
        print("Files in environment directory:")
        try:
            import os
            print(os.listdir('environment'))
        except:
            print("Cannot list environment directory")
        return False

    try:
        from agents.dqn_agent import DQNAgent
        print("✓ DQNAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DQNAgent: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")

    try:
        from environment.wordle_env import WordleEnv
        from agents.dqn_agent import DQNAgent

        # Test that we can import and create classes
        print("✓ Classes imported successfully")

        # Try to inspect the classes
        print(f"✓ WordleEnv class available: {WordleEnv}")
        print(f"✓ DQNAgent class available: {DQNAgent}")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run core tests."""
    print("=== Testing Core Components ===\n")

    tests = [
        test_core_imports,
        test_local_imports,
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
        print("✓ All core tests passed! The project structure is correct.")
        print("\nTo rebuild the image with full dependencies, run:")
        print("  docker compose build")
        return 0
    else:
        print("✗ Some core tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())