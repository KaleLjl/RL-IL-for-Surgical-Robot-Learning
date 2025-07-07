import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_tip_vs_eef():
    """Debug difference between TIP and EEF positions"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== TIP vs EEF POSITION DEBUG ===")
    
    psm = env.unwrapped.psm1
    
    for step in range(10):
        # Get TIP and EEF positions
        pos_tip, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
        pos_eef, _ = get_link_pose(psm.body, psm.EEF_LINK_INDEX)
        
        # Get from observation (should be EEF)
        obs_pos = obs['observation'][:3]
        
        print(f"Step {step}:")
        print(f"  TIP position: [{pos_tip[0]:.4f}, {pos_tip[1]:.4f}, {pos_tip[2]:.4f}]")
        print(f"  EEF position: [{pos_eef[0]:.4f}, {pos_eef[1]:.4f}, {pos_eef[2]:.4f}]")
        print(f"  OBS position: [{obs_pos[0]:.4f}, {obs_pos[1]:.4f}, {obs_pos[2]:.4f}]")
        
        # Calculate differences
        tip_eef_diff = np.linalg.norm(np.array(pos_tip) - np.array(pos_eef))
        obs_eef_diff = np.linalg.norm(np.array(obs_pos) - np.array(pos_eef))
        obs_tip_diff = np.linalg.norm(np.array(obs_pos) - np.array(pos_tip))
        
        print(f"  TIP-EEF distance: {tip_eef_diff:.6f}")
        print(f"  OBS-EEF distance: {obs_eef_diff:.6f}")
        print(f"  OBS-TIP distance: {obs_tip_diff:.6f}")
        print()
        
        action = env.unwrapped.get_oracle_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()

if __name__ == "__main__":
    debug_tip_vs_eef()