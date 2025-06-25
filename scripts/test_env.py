# Author(s): Lele Chen
# Created on: 2025-06-24
# Last modified: 2025-06-24

"""
A simple script to test the dvrk_gym environment.
This script will:
1. Install the dvrk_gym package in editable mode.
2. Create an instance of the NeedleReach-v0 environment.
3. Reset the environment.
4. Run a few random actions.
5. Print the observations.
"""
import gymnasium as gym
import numpy as np
import os
import subprocess

def main():
    # Install the package in editable mode
    print("Installing dvrk_gym package...")
    subprocess.run(["pip", "install", "-e", "."], check=True)
    print("Package installed.")

    # Import the package to ensure registration
    import dvrk_gym

    # Create the environment
    print("Creating environment 'dvrk_gym/NeedleReach-v0'...")
    env = gym.make('dvrk_gym/NeedleReach-v0', render_mode='human')
    print("Environment created.")

    # Reset the environment
    print("Resetting environment...")
    obs, info = env.reset()
    print("Initial observation:", obs)

    # Run a few random steps
    for i in range(100):
        action = env.action_space.sample()
        print(f"\n--- Step {i+1} ---")
        print(f"Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()

    print("\nTest finished. Press Enter in the terminal to close the window.")
    input()  # This will pause the script until you press Enter
    env.close()


if __name__ == "__main__":
    main()
