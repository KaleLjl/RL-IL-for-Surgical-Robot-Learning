åimport numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from dvrk_gym.envs.needle_reach import NeedleReachEnv

# This script is a Minimal, Reproducible Example (MRE) to diagnose
# a critical bug in the PPO agent.
#
# New Hypothesis: The policy is not ignoring the goal entirely, but its influence
# is dwarfed by the robot's own state ('observation'). This script tests this
# by using a REAL observation from the environment and only changing the goal.

# --- 1. Setup ---
# Load the trained model
MODEL_PATH = "logs/ppo_needle_reach_1750928921/run_1/checkpoints/rl_model_100000_steps.zip"
print(f"Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

# Create a real environment to get a realistic observation
print("\nCreating NeedleReachEnv to get a real observation...")
env = NeedleReachEnv(render_mode=None)
real_obs, info = env.reset(seed=42) # Use a seed for reproducibility
env.close() # We only need the single observation
print("Environment created and a real observation was sampled.")


# --- 2. Create Two Distinct Observations Based on Reality ---
# Observation 1: The real observation from the environment
obs1 = real_obs
# Ensure the goal is what we expect from the seed
# (This is just a sanity check, the env sets the goal)
print(f"\nOriginal goal from env reset: {obs1['desired_goal']}")


# Observation 2: Same as obs1, but with a manually changed goal
obs2 = obs1.copy()
obs2['desired_goal'] = np.array([-0.1, 0.1, -0.1], dtype=np.float32)


print("\nCreated two observations with identical REAL robot states but different goals.")
print(f"Observation 1 Goal: {obs1['desired_goal']}")
print(f"Observation 2 Goal: {obs2['desired_goal']}")


# --- 3. Predict Actions ---
# Get the model's predicted action for each observation.
# We set deterministic=True to get the greedy action without any exploration noise.
action1, _ = model.predict(obs1, deterministic=True)
action2, _ = model.predict(obs2, deterministic=True)

print("\nPredicted actions (deterministic):")
print(f"Action for Goal 1: {action1}")
print(f"Action for Goal 2: {action2}")

# --- 4. Assert and Conclude ---
# The core of the test: assert whether the actions are different.
# If the actions are the same, our hypothesis is confirmed.
are_actions_different = not np.allclose(action1, action2, atol=1e-6)

print("\n--- Diagnosis Result ---")
if are_actions_different:
    print("✅ SUCCESS: The actions are different.")
    print("This suggests the policy IS using the goal information.")
else:
    print("❌ FAILURE: The actions are identical.")
    print("This CONFIRMS the hypothesis: the policy is ignoring the 'desired_goal' input.")

# We add a formal assertion to make this script usable in automated checks.
assert are_actions_different, "Hypothesis Confirmed: Policy produces identical actions for different goals."

print("\nScript finished.")
