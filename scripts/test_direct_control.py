import gymnasium as gym
import numpy as np
import time
import dvrk_gym  # Import to register the environment

def test_direct_control(env_name, num_steps=200):
    """
    Tests the low-level robot control by sending a constant action.

    Args:
        env_name (str): The name of the Gymnasium environment.
        num_steps (int): The number of steps to run the test for.
    """
    print(f"Initializing environment: {env_name}")
    # Use 'human' render_mode to visualize the robot's movement
    env = gym.make(env_name, render_mode='human')

    print("Resetting environment...")
    obs, info = env.reset()

    # Define a constant action to move the robot in the positive x-direction.
    # This bypasses the get_oracle_action() logic entirely.
    # action format: (dx, dy, dz, d_roll, d_pitch, d_yaw)
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    print(f"Applying constant action for {num_steps} steps: {action}")

    for i in range(num_steps):
        print(f"Step {i + 1}/{num_steps}")
        obs, reward, terminated, truncated, info = env.step(action)
        
        # A small delay to make the visualization easier to follow.
        time.sleep(0.05)

        if terminated or truncated:
            print("Episode finished prematurely.")
            break
    
    print("\nTest finished.")
    env.close()

if __name__ == "__main__":
    ENV_NAME = "NeedleReach-v0"
    test_direct_control(ENV_NAME)
