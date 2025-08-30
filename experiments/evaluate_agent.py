import argparse
import os
import json
from typing import Tuple, List

import gymnasium as gym
import numpy as np
import imageio
import pybullet as p

# Algo loaders
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Register dvrk_gym envs and wrappers
import dvrk_gym
from dvrk_gym.utils.wrappers import FlattenDictObsWrapper


def detect_env_name(model_path: str) -> str:
    lp = model_path.lower()
    if "needle_reach" in lp or "needle" in lp:
        return "NeedleReach-v0"
    if "peg_transfer" in lp or "peg" in lp:
        return "PegTransfer-v0"
    # Default
    return "NeedleReach-v0"


def detect_algo(model_path: str, override: str = None) -> str:
    if override:
        return override
    lp = model_path.lower()
    if "ppo_il" in lp or "dapg" in lp or "ppo_bc" in lp:
        return "ppo_il"
    if "bc" in lp and not ("ppo" in lp and "final" in lp):
        return "bc"
    return "ppo"


def get_custom_camera_frame(env):
    scaling = getattr(env.unwrapped, 'SCALING', 1.0)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(-0.05 * scaling, 0, 0.375 * scaling),
        distance=1.2 * scaling,
        yaw=90,
        pitch=-25,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=45,
        aspect=16/9,
        nearVal=0.1,
        farVal=20.0
    )
    width, height = 1280, 720
    camera_data = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = camera_data[2]
    rgb_array = np.array(rgb_array).reshape(height, width, 4)[:, :, :3]
    return rgb_array


def flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate([
        obs['observation'],
        obs['achieved_goal'],
        obs['desired_goal']
    ])


def evaluate_standard(
    algo: str,
    model,
    env,
    episodes: int,
    save_video: bool = False,
    video_path: str = None,
) -> Tuple[float, float, List[float], List[float]]:
    episode_rewards: List[float] = []
    episode_successes: List[float] = []
    video_frames = []

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            if algo == 'bc':
                action, _ = model.predict(flatten_obs(obs), deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if save_video:
                video_frames.append(get_custom_camera_frame(env))

        episode_rewards.append(float(total_reward))
        episode_successes.append(float(info.get('is_success', 0.0)))

    if save_video and video_frames and video_path:
        imageio.mimsave(video_path, video_frames, fps=30)

    success_rate = float(np.mean(episode_successes) * 100.0) if episode_successes else 0.0
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    return success_rate, avg_reward, episode_rewards, episode_successes


def evaluate_with_action_noise(
    algo: str,
    model,
    env,
    episodes: int,
    noise_alpha: float,
) -> Tuple[float, float, List[float], List[float]]:
    # Relative, range-aware Gaussian noise: sigma = alpha * (range/2)
    low, high = env.action_space.low, env.action_space.high
    scale = (high - low) / 2.0

    episode_rewards: List[float] = []
    episode_successes: List[float] = []

    for i in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            if algo == 'bc':
                action, _ = model.predict(flatten_obs(obs), deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            action_noise = np.random.normal(0, noise_alpha * scale, action.shape)
            noisy_action = np.clip(action + action_noise, low, high)

            obs, reward, done, truncated, info = env.step(noisy_action)
            total_reward += reward

        episode_rewards.append(float(total_reward))
        episode_successes.append(float(info.get('is_success', 0.0)))

    success_rate = float(np.mean(episode_successes) * 100.0) if episode_successes else 0.0
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    return success_rate, avg_reward, episode_rewards, episode_successes


def main():
    parser = argparse.ArgumentParser(description="Unified evaluator for PPO, PPO+IL (DAPG), and BC")
    parser.add_argument("--model", required=True, help="Path to the trained model (.zip)")
    parser.add_argument("--output-dir", default=None, help="Directory to save evaluation results")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering for faster evaluation")
    parser.add_argument("--action-noise-test", action="store_true", help="Run noise-only evaluation")
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Relative noise alpha in [0,1]; sigma = alpha * (action_range/2)")
    parser.add_argument("--save-video", action="store_true", help="Save combined evaluation video")
    parser.add_argument("--video-dir", default="videos", help="Directory for saved videos")
    parser.add_argument("--algo", choices=["ppo", "ppo_il", "bc"], default=None,
                        help="Override algorithm detection")

    args = parser.parse_args()

    env_name = detect_env_name(args.model)
    algo = detect_algo(args.model, args.algo)

    # Render mode
    if args.save_video:
        render_mode = "rgb_array"
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "human"

    # Env kwargs: PPO uses dense rewards; others sparse
    env_kwargs = {"render_mode": render_mode}
    if algo == 'ppo':
        env_kwargs["use_dense_reward"] = True

    env = gym.make(env_name, **env_kwargs)

    # Observation handling
    wrapped_env = env
    if algo in ('ppo', 'ppo_il'):
        wrapped_env = FlattenDictObsWrapper(env)

    # Video setup
    video_path = None
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        sub = "ppo" if algo == 'ppo' else ("ppo_il" if algo == 'ppo_il' else "bc")
        folder = os.path.join(args.video_dir, f"{sub}_evaluation")
        os.makedirs(folder, exist_ok=True)
        video_path = os.path.join(folder, f"{sub}_eval_combined_all_episodes.mp4")

    # Load model
    try:
        if algo == 'bc':
            model = ActorCriticPolicy.load(args.model)
        else:
            model = PPO.load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    # Run evaluation
    if args.action_noise_test:
        print(f"--- Running noise-only evaluation (alpha={args.noise_std}) ---")
        success_rate, avg_reward, episode_rewards, episode_successes = evaluate_with_action_noise(
            algo, model, wrapped_env, args.episodes, args.noise_std
        )
    else:
        print("--- Running standard evaluation ---")
        success_rate, avg_reward, episode_rewards, episode_successes = evaluate_standard(
            algo, model, wrapped_env, args.episodes, args.save_video, video_path
        )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Algo: {algo}")
    print(f"Env: {env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average reward: {avg_reward:.3f}")
    if args.action_noise_test:
        print(f"Action noise alpha: {args.noise_std}")
    print("=" * 50)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Try to collect minimal hyperparams for provenance
        model_hparams = {}
        try:
            if algo != 'bc':
                # PPO attributes if available
                if hasattr(model, 'learning_rate'):
                    model_hparams['learning_rate'] = float(model.learning_rate)
                if hasattr(model, 'n_steps'):
                    model_hparams['n_steps'] = int(model.n_steps)
                if hasattr(model, 'batch_size'):
                    model_hparams['batch_size'] = int(model.batch_size)
                if hasattr(model, 'gamma'):
                    model_hparams['gamma'] = float(model.gamma)
                if hasattr(model, 'gae_lambda'):
                    model_hparams['gae_lambda'] = float(model.gae_lambda)
                if hasattr(model, 'clip_range'):
                    model_hparams['clip_range'] = float(model.clip_range)
            else:
                # BC saved as ActorCriticPolicy; net arch not always accessible
                if hasattr(model, 'net_arch'):
                    model_hparams['net_arch'] = model.net_arch
        except Exception:
            model_hparams = {}

        # Common payload
        results = {
            "model_path": args.model,
            "environment": env_name,
            "algorithm": "PPO" if algo == 'ppo' else ("PPO+IL" if algo == 'ppo_il' else "BC"),
            "total_episodes": int(args.episodes),
            "average_reward": float(avg_reward),
            "success_rate": float(success_rate),
            "episode_rewards": [float(r) for r in episode_rewards],
            "episode_successes": [float(s) for s in episode_successes],
            "hyperparameters": {
                "model_hyperparams": model_hparams,
                "reward_type": "dense" if algo == 'ppo' else "sparse",
                "observation_space": "flattened_dict" if algo != 'bc' else "dict_manual_flatten"
            },
        }

        # Marker for noise mode
        results["action_noise_test"] = {
            "enabled": bool(args.action_noise_test),
            "noise_std": float(args.noise_std) if args.action_noise_test else None,
        }

        # Filename per algo for backward compatibility
        if algo == 'ppo':
            fname = "ppo_evaluation_results.json"
        elif algo == 'ppo_il':
            fname = "ppo_il_evaluation_results.json"
        else:
            fname = "evaluation_results.json"

        out_path = os.path.join(args.output_dir, fname)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {out_path}")

    wrapped_env.close()


if __name__ == "__main__":
    main()

