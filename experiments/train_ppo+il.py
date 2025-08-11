import os
import pickle
import time
import gymnasium as gym
import numpy as np
import torch
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy as MlpPolicy
from stable_baselines3.common.callbacks import CallbackList
from imitation.data import types
from imitation.util import logger as imitation_logger
from stable_baselines3.common import logger as sb3_logger

# Import our custom algorithm and wrapper
from dvrk_gym.algorithms.ppo_bc import PPOWithBCLoss
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper
from dvrk_gym.utils.callbacks import TrainingAnalysisCallback
import dvrk_gym  # Import to register the environment

def extract_network_architecture(bc_model_path):
    """
    Extract network architecture from a trained BC model.
    
    Args:
        bc_model_path (str): Path to the BC model file
        
    Returns:
        list: Network architecture as a list of hidden layer sizes
    """
    try:
        import torch
        
        # Load the BC policy directly from the zip file
        bc_policy = MlpPolicy.load(bc_model_path)
        
        # Try different ways to extract architecture
        net_arch = []
        
        # Method 1: Check mlp_extractor.shared_net
        if hasattr(bc_policy, 'mlp_extractor') and hasattr(bc_policy.mlp_extractor, 'shared_net'):
            for layer in bc_policy.mlp_extractor.shared_net:
                if isinstance(layer, torch.nn.Linear):
                    net_arch.append(layer.out_features)
            
        # Method 2: Check policy_net if shared_net doesn't exist
        elif hasattr(bc_policy, 'mlp_extractor') and hasattr(bc_policy.mlp_extractor, 'policy_net'):
            for layer in bc_policy.mlp_extractor.policy_net:
                if isinstance(layer, torch.nn.Linear):
                    net_arch.append(layer.out_features)
        
        # Method 3: Check action_net directly
        elif hasattr(bc_policy, 'action_net'):
            for layer in bc_policy.action_net:
                if isinstance(layer, torch.nn.Linear):
                    net_arch.append(layer.out_features)
        
        # Method 4: Inspect all modules recursively
        else:
            for name, module in bc_policy.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if 'shared' in name or 'policy' in name or 'mlp' in name:
                        net_arch.append(module.out_features)
        
        # Return detected architecture
        if net_arch:
            print(f"Detected BC model architecture: {net_arch}")
            return net_arch
        else:
            print("Could not detect network architecture from BC model, using default [256, 256]")
            return [256, 256]
            
    except Exception as e:
        print(f"Failed to extract architecture from BC model: {e}")
        print("Using default architecture [256, 256]")
        return [256, 256]

def train_dapg_agent(env_name, expert_data_path, model_save_path=None, log_dir=None, timesteps=300000, bc_weight=0.05, log_interval=1000, bc_model_path=None, render=False):
    """
    Trains an agent using our custom PPOWithBCLoss algorithm, which
    forms the basis of our DAPG implementation.

    Args:
        env_name (str): The name of the Gymnasium environment.
        expert_data_path (str): Path to the expert demonstration data (.pkl file).
        model_save_path (str): Path to save the trained policy model.
        log_dir (str): Directory to save training logs.
        timesteps (int): Total training timesteps.
        bc_weight (float): BC loss weight.
        log_interval (int): Steps between analysis logging.
    """
    print("--- Custom DAPG Training ---")
    
    # --- 1. Load and Flatten Expert Data ---
    print(f"Loading expert data from: {expert_data_path}")
    with open(expert_data_path, "rb") as f:
        trajectories = pickle.load(f)

    print("Flattening Dict observations into a single array...")
    all_obs, all_next_obs, all_acts, all_dones = [], [], [], []
    for traj in trajectories:
        obs_soa = traj["obs"]
        # The last observation has no corresponding action, so we ignore it.
        num_transitions = len(traj["acts"])

        # Flatten each observation dictionary into a single numpy array
        for i in range(num_transitions):
            flat_obs = np.concatenate([
                obs_soa['observation'][i],
                obs_soa['achieved_goal'][i],
                obs_soa['desired_goal'][i]
            ])
            all_obs.append(flat_obs)
            
            # We need next_obs and dones for the Transitions object.
            flat_next_obs = np.concatenate([
                obs_soa['observation'][i+1],
                obs_soa['achieved_goal'][i+1],
                obs_soa['desired_goal'][i+1]
            ])
            all_next_obs.append(flat_next_obs)

        all_acts.extend(traj["acts"])
        # Create a dones array that is True only at the very end of the trajectory.
        dones = [False] * (num_transitions - 1) + [True]
        all_dones.extend(dones)

    # Convert to a format that can be passed to our custom algorithm
    expert_demonstrations = types.Transitions(
        obs=np.array(all_obs),
        acts=np.array(all_acts),
        next_obs=np.array(all_next_obs),
        dones=np.array(all_dones),
        infos=np.array([{} for _ in range(len(all_obs))]),
    )
    print(f"Data flattened and converted to Transitions format: {len(expert_demonstrations)} samples.")

    # --- 2. Setup Vectorized and Flattened Environment ---
    print(f"Initializing environment: {env_name}")
    # Per reward-system-guidelines.md, DAPG should use a sparse reward.
    # The default is sparse, so we don't need to set any env_kwargs.
    # Setup environment with optional rendering
    render_mode = 'human' if render else None
    venv = make_vec_env(
        env_name,
        n_envs=1,
        wrapper_class=FlattenDictObsWrapper,
        env_kwargs={'render_mode': render_mode}
    )
    print("Environment created and wrapped with FlattenDictObsWrapper.")
    if render:
        print("Rendering enabled - training will be visualized")
    else:
        print("Rendering disabled - training will run faster")

    # --- 3. Configure Logging ---
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        imitation_logger.configure(folder=log_dir, format_strs=["stdout", "tensorboard"])
        sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
        print(f"Logging configured at: {log_dir}")
    else:
        print("Logging disabled - no output directory specified")

    # --- 4. Setup Custom DAPG (PPOWithBCLoss) Trainer ---
    print("Initializing custom PPOWithBCLoss agent...")
    
    # Extract network architecture from BC model to ensure exact matching
    if bc_model_path and os.path.exists(bc_model_path):
        print("Extracting network architecture from BC model...")
        net_arch = extract_network_architecture(bc_model_path)
    else:
        print("BC model not found, using default architecture [256, 256]")
        net_arch = [256, 256]
    
    policy_kwargs = dict(
        net_arch=net_arch,  # Dynamically extracted from BC model
    )
    
    model = PPOWithBCLoss(
        policy=MlpPolicy,
        env=venv,
        expert_demonstrations=expert_demonstrations,
        bc_loss_weight=bc_weight,
        bc_batch_size=1024,  # Further increased for more stable BC loss
        tensorboard_log=log_dir,
        learning_rate=1e-4,  # Reduced LR for more stable value function learning
        n_steps=4096,  # Increased for better sample efficiency
        batch_size=256,  # Significantly increased to reduce reward variance
        n_epochs=5,  # Reduced epochs to prevent overfitting
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,  # Use same clip range for value function
        vf_coef=0.5,  # Standard value function coefficient
        max_grad_norm=0.5,  # Gradient clipping for stability
        ent_coef=0.0,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    
    # --- 4.5. Load BC model weights if provided (Standard DAPG approach) ---
    if bc_model_path and os.path.exists(bc_model_path):
        print(f"Loading BC model weights from: {bc_model_path}")
        try:
            # BC model was saved using imitation library's policy.save() method
            # We need to load it as a policy directly, not as a stable-baselines3 model
            from gymnasium.spaces import Box
            
            # Get the actual observation space from the environment
            dummy_env = venv.envs[0].unwrapped if hasattr(venv.envs[0], 'unwrapped') else venv.envs[0]
            obs_space = venv.observation_space
            action_space = venv.action_space
            
            # Load BC policy directly
            bc_policy = MlpPolicy.load(bc_model_path)
            
            # Copy BC policy weights to DAPG model with flexible matching
            try:
                model.policy.load_state_dict(bc_policy.state_dict())
                print("BC model weights loaded successfully (exact match).")
            except RuntimeError as e:
                print("Attempting flexible weight transfer...")
                
                # Get state dicts
                bc_state = bc_policy.state_dict()
                ppo_state = model.policy.state_dict()
                
                # Try to match and transfer compatible weights
                transferred = 0
                for ppo_key, ppo_tensor in ppo_state.items():
                    # Try exact match first
                    if ppo_key in bc_state and ppo_tensor.shape == bc_state[ppo_key].shape:
                        ppo_state[ppo_key] = bc_state[ppo_key].clone()
                        transferred += 1
                    else:
                        # Try to find similar keys (e.g., different network structure)
                        for bc_key, bc_tensor in bc_state.items():
                            if ppo_tensor.shape == bc_tensor.shape:
                                # Check if keys are semantically similar
                                if ('policy' in ppo_key and 'policy' in bc_key) or \
                                   ('value' in ppo_key and 'value' in bc_key) or \
                                   ('action' in ppo_key and 'action' in bc_key):
                                    ppo_state[ppo_key] = bc_tensor.clone()
                                    transferred += 1
                                    break
                
                # Load the updated state dict
                model.policy.load_state_dict(ppo_state)
                print(f"BC model weights transferred: {transferred} layers")
        except Exception as e:
            print(f"Failed to load BC model: {e}")
            print("Starting from random initialization.")
    else:
        print("No BC model provided - starting from random initialization.")
    
    print("Custom agent configured.")

    # --- 4.5. Setup Training Analysis Callback ---
    analysis_callback = TrainingAnalysisCallback(
        log_interval=log_interval,
        analysis_dir=os.path.join(log_dir, "analysis"),
        algorithm="dapg",
        verbose=1
    )
    print("Training analysis callback configured.")

    # --- 5. Train the Agent ---
    print("Starting training...")
    model.learn(total_timesteps=timesteps, callback=analysis_callback)
    print("Training complete.")

    # --- 6. Save the Model ---
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Trained model saved to: {model_save_path}")
    else:
        print("Model not saved - no output directory specified")

    venv.close()

if __name__ == "__main__":
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train PPO+IL (DAPG) agent on dVRK environments")
    parser.add_argument("--env", required=True,
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to train on")
    parser.add_argument("--bc-model", required=True,
                       help="Path to BC model for initialization")
    parser.add_argument("--expert-data", required=True,
                       help="Path to expert demonstration data (.pkl file)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for logs and model (if not specified, no files are saved)")
    parser.add_argument("--render", action="store_true",
                       help="Enable rendering during training (default: disabled)")
    
    args = parser.parse_args()
    
    # Environment-specific optimal parameters for PPO+IL
    if args.env == "NeedleReach-v0":
        timesteps = 300000
        bc_weight = 0.05
    elif args.env == "PegTransfer-v0":
        timesteps = 500000
        bc_weight = 0.02  # Further reduced BC weight to reduce over-reliance on BC
    
    print(f"Training PPO+IL on {args.env} with optimized parameters:")
    print(f"  Timesteps: {timesteps}")
    print(f"  BC weight: {bc_weight}")
    print(f"  BC model: {args.bc_model}")
    print(f"  Expert data: {args.expert_data}")
    
    # Use user-provided expert data path
    expert_data_path = args.expert_data
    
    # Set up output paths only if output_dir is specified
    if args.output_dir:
        env_suffix = args.env.lower().replace("-v0", "").replace("reach", "_reach").replace("transfer", "_transfer")
        experiment_name = f"ppo_il_{env_suffix}_{int(time.time())}"
        log_dir = os.path.join(args.output_dir, "logs")
        model_save_path = os.path.join(args.output_dir, f"{experiment_name}.zip")
        print(f"Output directory: {args.output_dir}")
    else:
        log_dir = None
        model_save_path = None
        print("No output directory specified - running without saving files")

    train_dapg_agent(
        env_name=args.env,
        expert_data_path=expert_data_path,
        model_save_path=model_save_path,
        log_dir=log_dir,
        timesteps=timesteps,
        bc_weight=bc_weight,
        log_interval=1000,
        bc_model_path=args.bc_model,
        render=args.render,
    )
