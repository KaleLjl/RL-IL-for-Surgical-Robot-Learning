import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p
from dvrk_gym.utils.pybullet_utils import get_link_pose

def debug_collision_config():
    """Debug gripper collision configuration"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== GRIPPER COLLISION CONFIGURATION DEBUG ===")
    
    psm = env.unwrapped.psm1
    obj_id = env.unwrapped.obj_id
    
    print(f"PSM body ID: {psm.body}")
    print(f"Object ID: {obj_id}")
    print()
    
    # Check gripper link details
    print("=== GRIPPER LINKS ANALYSIS ===")
    for link_idx in [6, 7, 8]:  # gripper1, gripper2, tip
        joint_info = p.getJointInfo(psm.body, link_idx)
        print(f"Link {link_idx} ({joint_info[12].decode('utf-8')}):")
        
        # Get collision shape info
        collision_info = p.getCollisionShapeData(psm.body, link_idx)
        print(f"  Collision shapes: {len(collision_info)}")
        for i, shape in enumerate(collision_info):
            print(f"    Shape {i}: type={shape[2]}, dimensions={shape[3]}")
        
        # Get visual shape info
        visual_info = p.getVisualShapeData(psm.body, link_idx)
        print(f"  Visual shapes: {len(visual_info)}")
        
        print()
    
    # Check object collision info
    print("=== OBJECT COLLISION ANALYSIS ===")
    obj_collision_info = p.getCollisionShapeData(obj_id, -1)
    print(f"Object collision shapes: {len(obj_collision_info)}")
    for i, shape in enumerate(obj_collision_info):
        print(f"  Shape {i}: type={shape[2]}, dimensions={shape[3]}")
    print()
    
    # Check current collision filter pairs
    print("=== COLLISION FILTER STATUS ===")
    print("Checking if collisions are disabled...")
    
    # Test collision detection manually
    print("=== MANUAL COLLISION TEST ===")
    for step in range(100):
        action = env.unwrapped.get_oracle_action(obs)
        
        if step > 20 and step % 10 == 0:  # Check every 10 steps after movement
            print(f"\nStep {step}:")
            
            # Get positions
            pos_tip, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
            pos_obj, _ = get_link_pose(obj_id, -1)
            distance = np.linalg.norm(np.array(pos_tip) - np.array(pos_obj))
            
            print(f"  TIP-Object distance: {distance:.6f}")
            
            # Check all contact points with the robot
            all_contacts = p.getContactPoints(bodyA=psm.body)
            print(f"  Total robot contacts: {len(all_contacts)}")
            
            # Check specific gripper contacts
            contacts_6 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
            contacts_7 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            
            print(f"  Link 6 contacts: {len(contacts_6)}")
            print(f"  Link 7 contacts: {len(contacts_7)}")
            
            # Check contacts specifically with our object
            obj_contacts_6 = [c for c in contacts_6 if c[2] == obj_id]
            obj_contacts_7 = [c for c in contacts_7 if c[2] == obj_id]
            
            print(f"  Link 6 → Object contacts: {len(obj_contacts_6)}")
            print(f"  Link 7 → Object contacts: {len(obj_contacts_7)}")
            
            # Check collision filter status
            try:
                # This is a hack to check if collision is enabled
                # We can't directly query collision filter status in PyBullet
                print(f"  Gripper action: {action[4]:.2f}")
            except:
                pass
            
            if distance < 0.02:  # When very close
                print("  >>> VERY CLOSE - Should have contacts if collision works!")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    env.close()

if __name__ == "__main__":
    debug_collision_config()