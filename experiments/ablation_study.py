import os
import pickle
import time
import json
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

def train_ppo_bc_with_alpha(env_name, expert_data_path, alpha, model_save_path=None, 
                            log_dir=None, timesteps=300000, log_interval=1000, 
                            bc_model_path=None, render=False):
    """
    Trains an agent using PPOWithBCLoss with a specific alpha (BC weight).
    
    Args:
        env_name (str): The name of the Gymnasium environment.
        expert_data_path (str): Path to the expert demonstration data (.pkl file).
        alpha (float): BC loss weight (0.0 = pure PPO, 1.0 = pure BC).
        model_save_path (str): Path to save the trained policy model.
        log_dir (str): Directory to save training logs.
        timesteps (int): Total training timesteps.
        log_interval (int): Steps between analysis logging.
        bc_model_path (str): Path to BC model for initialization.
        render (bool): Whether to render during training.
    
    Returns:
        dict: Training results including final success rate and other metrics.
    """
    print(f"\n--- Training PPO+BC with alpha={alpha} ---")
    
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
    # Setup environment with optional rendering
    render_mode = 'human' if render else None
    venv = make_vec_env(
        env_name,
        n_envs=1,
        wrapper_class=FlattenDictObsWrapper,
        env_kwargs={'render_mode': render_mode}
    )
    print("Environment created and wrapped with FlattenDictObsWrapper.")

    # --- 3. Configure Logging ---
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        imitation_logger.configure(folder=log_dir, format_strs=["stdout", "tensorboard"])
        sb3_logger.configure(folder=log_dir, format_strings=["stdout", "tensorboard"])
        print(f"Logging configured at: {log_dir}")
    else:
        print("Logging disabled - no output directory specified")

    # --- 4. Setup PPOWithBCLoss Agent ---
    print(f"Initializing PPOWithBCLoss agent with alpha={alpha}...")
    
    # Environment-specific hyperparameters
    if env_name == "NeedleReach-v0":
        # NeedleReach optimal parameters
        default_net_arch = [256, 256]
        learning_rate = 1e-4
        n_steps = 4096
        batch_size = 256
        n_epochs = 5
        bc_batch_size = 1024
    elif env_name == "PegTransfer-v0":
        # PegTransfer optimal parameters (smaller network, adjusted hyperparams)
        default_net_arch = [128, 128]
        learning_rate = 1e-4
        n_steps = 4096
        batch_size = 256
        n_epochs = 5
        bc_batch_size = 1024
    else:
        # Fallback to default parameters
        default_net_arch = [256, 256]
        learning_rate = 1e-4
        n_steps = 4096
        batch_size = 256
        n_epochs = 5
        bc_batch_size = 1024
    
    # Extract network architecture from BC model to ensure exact matching
    if bc_model_path and os.path.exists(bc_model_path):
        print("Extracting network architecture from BC model...")
        net_arch = extract_network_architecture(bc_model_path)
    else:
        print(f"BC model not found, using environment-specific architecture {default_net_arch}")
        net_arch = default_net_arch
    
    policy_kwargs = dict(
        net_arch=net_arch,
    )
    
    # Create model with specified alpha (bc_loss_weight)
    model = PPOWithBCLoss(
        policy=MlpPolicy,
        env=venv,
        expert_demonstrations=expert_demonstrations,
        bc_loss_weight=alpha,  # Use alpha parameter here
        bc_batch_size=bc_batch_size,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.0,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    
    # --- 5. Load BC model weights if provided and alpha > 0 ---
    if alpha > 0 and bc_model_path and os.path.exists(bc_model_path):
        print(f"Loading BC model weights from: {bc_model_path}")
        try:
            # Load BC policy directly
            bc_policy = MlpPolicy.load(bc_model_path)
            
            # Copy BC policy weights to PPO+BC model
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
                    if ppo_key in bc_state and ppo_tensor.shape == bc_state[ppo_key].shape:
                        ppo_state[ppo_key] = bc_state[ppo_key].clone()
                        transferred += 1
                
                # Load the updated state dict
                model.policy.load_state_dict(ppo_state)
                print(f"BC model weights transferred: {transferred} layers")
        except Exception as e:
            print(f"Failed to load BC model: {e}")
            print("Starting from random initialization.")
    else:
        if alpha == 0:
            print("Pure PPO (alpha=0) - starting from random initialization.")
        else:
            print("No BC model provided - starting from random initialization.")
    
    print("Agent configured.")

    # --- 6. Setup Training Analysis Callback ---
    analysis_callback = TrainingAnalysisCallback(
        log_interval=log_interval,
        analysis_dir=os.path.join(log_dir, "analysis") if log_dir else None,
        algorithm=f"ppo_bc_alpha_{alpha}",
        verbose=1
    )
    print("Training analysis callback configured.")

    # --- 7. Train the Agent ---
    print(f"Starting training with alpha={alpha}...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=analysis_callback)
    training_time = time.time() - start_time
    print(f"Training complete. Time: {training_time:.2f} seconds")

    # --- 8. Extract Success Rate from Analysis Files ---
    print("Extracting success rate from training analysis...")
    
    # Get the latest analysis file
    analysis_dir = os.path.join(log_dir, "analysis") if log_dir else None
    success_rate = 0.0
    mean_reward = 0.0
    std_reward = 0.0
    
    if analysis_dir and os.path.exists(analysis_dir):
        try:
            # Find the latest analysis file
            analysis_files = [f for f in os.listdir(analysis_dir) if f.startswith('analysis_') and f.endswith('.json')]
            if analysis_files:
                # Sort by timestep (extract number from filename: analysis_0001_step_1000.json)
                latest_file = max(analysis_files, key=lambda x: int(x.split('_')[3].split('.')[0]))
                analysis_path = os.path.join(analysis_dir, latest_file)
                
                # Read the analysis data
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
                
                # Extract success rate and reward metrics
                success_rate = analysis_data['metrics']['success_rate']['rate']
                mean_reward = analysis_data['metrics']['episode_reward']['mean']
                std_reward = analysis_data['metrics']['episode_reward']['std']
                
                print(f"Success rate from analysis: {success_rate:.1%}")
            else:
                print("No analysis files found, falling back to policy evaluation...")
                # Fallback to standard evaluation
                from stable_baselines3.common.evaluation import evaluate_policy
                mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=100, deterministic=True)
                success_rate = max(0.0, mean_reward)  # Rough approximation
        except Exception as e:
            print(f"Error reading analysis files: {e}")
            print("Falling back to policy evaluation...")
            # Fallback to standard evaluation
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=100, deterministic=True)
            success_rate = max(0.0, mean_reward)  # Rough approximation
    else:
        print("No analysis directory found, using policy evaluation...")
        # Fallback to standard evaluation
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=100, deterministic=True)
        success_rate = max(0.0, mean_reward)  # Rough approximation
    
    results = {
        "alpha": alpha,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "training_time": training_time,
        "timesteps": timesteps
    }
    
    print(f"Evaluation Results for alpha={alpha}:")
    print(f"  Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  Success rate: {success_rate:.1%}")

    # --- 9. Save the Model ---
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Trained model saved to: {model_save_path}")
    else:
        print("Model not saved - no output path specified")

    venv.close()
    
    return results

def run_ablation_study(env_name, expert_data_path, bc_model_path, output_dir, 
                       alphas=None, timesteps=None, render=False):
    """
    Run full ablation study with multiple alpha values.
    
    Args:
        env_name (str): Environment name.
        expert_data_path (str): Path to expert demonstrations.
        bc_model_path (str): Path to BC model for initialization.
        output_dir (str): Base output directory for all results.
        alphas (list): List of alpha values to test (if None, uses env-specific defaults).
        timesteps (int): Training timesteps per run (if None, uses env-specific defaults).
        render (bool): Whether to render during training.
    
    Returns:
        dict: Results from all runs.
    """
    # Use environment-specific defaults if alphas not specified
    if alphas is None:
        # Use same alpha values for both environments for consistency
        alphas = [0.0, 0.05, 0.2, 0.5, 1.0]
    
    # Use environment-specific defaults if timesteps not specified
    if timesteps is None:
        if env_name == "NeedleReach-v0":
            timesteps = 300000
        elif env_name == "PegTransfer-v0":
            timesteps = 500000  # or 700000 based on train_ppo+bc.py
        else:
            timesteps = 300000  # default fallback
    
    print("=" * 60)
    print(f"ABLATION STUDY: PPO+BC on {env_name}")
    print(f"Alpha values: {alphas}")
    print(f"Timesteps per run: {timesteps}")
    print("=" * 60)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Running alpha={alpha}")
        print(f"{'='*60}")
        
        # Create unique output paths
        run_name = f"alpha_{alpha}"
        model_path = os.path.join(output_dir, "models", f"{run_name}.zip")
        log_dir = os.path.join(output_dir, "logs", run_name)
        
        # Train model
        results = train_ppo_bc_with_alpha(
            env_name=env_name,
            expert_data_path=expert_data_path,
            alpha=alpha,
            model_save_path=model_path,
            log_dir=log_dir,
            timesteps=timesteps,
            bc_model_path=bc_model_path if alpha > 0 else None,
            render=render
        )
        
        all_results.append(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE - SUMMARY")
    print("=" * 60)
    
    for result in all_results:
        alpha = result["alpha"]
        success = result["success_rate"]
        print(f"Alpha {alpha:4.2f}: Success Rate = {success:6.1%}")
    
    
    return all_results

if __name__ == "__main__":
    # --- Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Ablation study for PPO+BC on dVRK environments")
    parser.add_argument("--env", required=True,
                       choices=["NeedleReach-v0", "PegTransfer-v0"],
                       help="Environment name to train on")
    parser.add_argument("--bc-model", required=True,
                       help="Path to BC model for initialization (used when alpha > 0)")
    parser.add_argument("--expert-data", required=True,
                       help="Path to expert demonstration data (.pkl file)")
    parser.add_argument("--output-dir", required=True,
                       help="Base output directory for all ablation study results")
    parser.add_argument("--alphas", nargs='+', type=float,
                       default=None,
                       help="Alpha values (BC weights) to test (default: [0.0, 0.05, 0.2, 0.5, 1.0] for both environments)")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Training timesteps per run (default: env-specific - 300k for NeedleReach, 500k for PegTransfer)")
    parser.add_argument("--render", action="store_true",
                       help="Enable rendering during training (default: disabled)")
    
    args = parser.parse_args()
    
    # Display configuration with defaults
    if args.alphas is None:
        display_alphas = [0.0, 0.05, 0.2, 0.5, 1.0]
    else:
        display_alphas = args.alphas
    
    if args.timesteps is None:
        display_timesteps = 300000 if args.env == "NeedleReach-v0" else 500000
    else:
        display_timesteps = args.timesteps
    
    print(f"Starting ablation study for PPO+BC on {args.env}")
    print(f"Configuration:")
    print(f"  Alpha values: {display_alphas}")
    print(f"  Timesteps: {display_timesteps}")
    print(f"  BC model: {args.bc_model}")
    print(f"  Expert data: {args.expert_data}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run ablation study
    results = run_ablation_study(
        env_name=args.env,
        expert_data_path=args.expert_data,
        bc_model_path=args.bc_model,
        output_dir=args.output_dir,
        alphas=args.alphas,
        timesteps=args.timesteps,
        render=args.render
    )
    
    print("\nAblation study complete!")