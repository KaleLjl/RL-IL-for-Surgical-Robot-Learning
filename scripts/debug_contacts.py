import gymnasium as gym
import numpy as np
import dvrk_gym
import pybullet as p

def debug_contacts():
    """Debug contact detection in detail"""
    env = gym.make('PegTransfer-v0', render_mode='human')
    obs, info = env.reset()
    
    print("=== CONTACT DETECTION DEBUGGING ===")
    
    # Get gripper info
    psm = env.unwrapped.psm1
    print(f"PSM body ID: {psm.body}")
    print(f"EEF link index: {psm.EEF_LINK_INDEX}")
    print(f"TIP link index: {psm.TIP_LINK_INDEX}")
    print(f"Object ID: {env.unwrapped.obj_id}")
    
    # Check gripper links
    for i in range(p.getNumJoints(psm.body)):
        link_info = p.getJointInfo(psm.body, i)
        print(f"Link {i}: {link_info[12].decode('utf-8')}")
    
    print("\n--- Starting movement ---")
    
    for step in range(100):
        action = env.unwrapped.get_oracle_action(obs)
        
        # Check all contact points involving the robot
        all_contacts = p.getContactPoints(bodyA=psm.body)
        if all_contacts:
            print(f"Step {step}: Total contacts with robot: {len(all_contacts)}")
            for contact in all_contacts:
                if contact[2] == env.unwrapped.obj_id:  # Contact with target object
                    print(f"  Contact with object: linkA={contact[3]}, linkB={contact[4]}")
        
        # Check specific gripper links
        if step > 20:  # After some movement
            contacts_6 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
            contacts_7 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            
            obj_contacts_6 = [c for c in contacts_6 if c[2] == env.unwrapped.obj_id]
            obj_contacts_7 = [c for c in contacts_7 if c[2] == env.unwrapped.obj_id]
            
            if obj_contacts_6 or obj_contacts_7:
                print(f"Step {step}: Gripper contacts with object:")
                print(f"  Link 6: {len(obj_contacts_6)} contacts")
                print(f"  Link 7: {len(obj_contacts_7)} contacts")
                print(f"  Gripper action: {action[4]:.2f}")
                print(f"  Activation status: {env.unwrapped._activated}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if env.unwrapped._activated >= 0:
            print(f"*** ACTIVATED at step {step}! ***")
            break
            
        if terminated or truncated:
            print(f"Episode ended at step {step} without activation")
            break
    
    env.close()

if __name__ == "__main__":
    debug_contacts()