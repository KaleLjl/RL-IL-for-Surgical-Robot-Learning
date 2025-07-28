# Complete Gripper State Management and Progressive Reward System for PegTransfer

## Overview

This document describes the complete gripper state management system with three-phase blocking and the progressive reward system implemented to solve the PPO training failure on the PegTransfer task.

## Problem Statement

The original penalty-based reward system failed because:
1. **Success rate: 0%** after 500k timesteps
2. Robot learned to approach objects but never attempted grasping
3. The penalty (-0.3) for failed grasps discouraged exploration
4. Insufficient reward gradient between approach and grasp phases

## Solution: Complete Gripper State Management + Progressive Rewards

### 1. Three-Phase Gripper Blocking System

**Implementation**: The gripper state is managed across three distinct phases:

```python
# In _set_action() method
is_grasped = self._activated >= 0 and self._contact_constraint is not None

if is_grasped:
    # TRANSPORT PHASE: Keep gripper closed until near goal
    dist_to_goal = np.linalg.norm(np.array(obj_pos) - self.goal)
    if dist_to_goal > 0.02 * self.SCALING:  # 2cm threshold
        self.block_gripper = True  # Force closed during transport
    else:
        self.block_gripper = False  # Allow release near goal
else:
    # APPROACH PHASE: Block gripper when far from object
    tip_pos, _ = get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)
    dist_to_obj = np.linalg.norm(np.array(tip_pos) - np.array(obj_pos))
    if dist_to_obj > 0.03 * self.SCALING:  # 3cm threshold
        self.block_gripper = True
    else:
        self.block_gripper = False  # GRASP PHASE: Allow grasping
```

**Three Phases**:
1. **Approach Phase**: Gripper blocked when >3cm from object
2. **Grasp Phase**: Gripper free when ≤3cm from object
3. **Transport Phase**: Gripper blocked until <2cm from goal

**Benefits**:
- Prevents reward hacking at all stages
- Enforces complete task sequence: approach → grasp → transport → release
- Prevents accidental drops during transport
- Mimics natural human grasping behavior

### 2. Progressive Reward System

**Key Features**:
- No penalties for failed attempts (encourages exploration)
- No grasp attempt bonuses (prevents reward hacking)
- Clear reward progression that incentivizes task completion

**Reward Structure**:

#### Pre-Grasp Phase (approach only)
- **Far (>20cm)**: 0.0 to 0.3 (distance gradient)
- **Medium (5cm)**: 0.5
- **Close (3cm)**: 1.0 (gripper unlocks here)
- **Very close (1.5cm)**: 1.5

#### Post-Grasp Phase
- **Successful grasp**: 5.0 (major jump from pre-grasp max of 1.5)
- **Transport progress**: 0.0 to 5.0 (based on progress to goal)
- **Task completion**: 10.0

### 3. Why This Works

1. **Clear Incentive Gradient**:
   - Approach rewards (0-1.5) < Grasp reward (5.0) < Transport (5-10)
   - Robot learns that grasping is much more valuable than just approaching

2. **No Exploitation Paths**:
   - Can't spam gripper when far away (blocked)
   - No bonuses for failed grasp attempts
   - Must actually grasp to get high rewards

3. **Natural Learning Progression**:
   - First learns to approach (easy, immediate rewards)
   - Then discovers grasping when close (big reward jump)
   - Finally learns to transport (additional rewards)

## Implementation Details

### Files Modified
- `src/dvrk_gym/envs/peg_transfer.py`:
  - `_set_action()`: Added three-phase gripper blocking logic
  - `_get_dense_reward()`: Implemented progressive reward system

### Key Parameters
- **Approach phase**: Gripper blocked when >3cm from object
- **Grasp phase**: Gripper free when ≤3cm from object  
- **Transport phase**: Gripper blocked until <2cm from goal
- **Approach reward range**: 0.0 - 1.5
- **Grasp reward**: 5.0
- **Transport reward**: 0.0 - 5.0
- **Success reward**: 10.0

## Testing

Run `scripts/test_gripper_blocking.py` to verify:
1. Approach phase: Gripper blocking when far from object
2. Grasp phase: Gripper unlocking when close to object
3. Transport phase: Gripper stays closed during transport
4. Release phase: Gripper can open when near goal
5. Correct reward values at different distances
6. Complete integration with oracle policy

## Expected Training Improvements

With this complete system, PPO should:
1. Quickly learn to approach objects (clear gradient, no gripper distractions)
2. Discover grasping once close (large reward jump, gripper now available)
3. Learn to maintain grasp during transport (gripper locked, prevents drops)
4. Complete full task sequence reliably (enforced by state management)
5. Achieve >50% success rate within 500k timesteps

## Usage

No changes needed to training scripts. The environment automatically uses:
- Gripper blocking in all modes
- Progressive rewards when `use_dense_reward=True`
- Original sparse rewards when `use_dense_reward=False`

## Future Enhancements

1. **Adaptive thresholds**: Could adjust gripper unlock distance based on object size
2. **Curriculum learning**: Start with larger unlock radius, gradually decrease
3. **Different thresholds**: For different objects or task phases