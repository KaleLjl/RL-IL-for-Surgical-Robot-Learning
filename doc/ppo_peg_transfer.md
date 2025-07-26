# PPO Implementation for PegTransfer Task

## Problem Description

### Overview
The PegTransfer task requires a robotic arm to grasp a block and place it on a target peg. While conceptually simple, implementing PPO (Proximal Policy Optimization) for this task presents significant challenges related to reward engineering and credit assignment.

### Why Reward Systems Matter

1. **Sparse vs Dense Dilemma**
   - **Sparse rewards** (0/-1 for success/failure) are clean but provide no learning signal during exploration
   - **Dense rewards** provide continuous feedback but are prone to reward hacking
   - PPO from scratch requires dense rewards to learn, but dense rewards often lead to local optima

2. **Temporal Credit Assignment**
   - PegTransfer involves a sequence: approach → grasp → lift → transport → place
   - Each sub-task depends on the previous one's success
   - Rewarding early stages without ensuring later success leads to suboptimal policies

3. **The Value Function Problem**
   - PPO's value function learns to predict future rewards
   - If rewards are poorly designed, the value function learns incorrect associations
   - High value loss (>100) indicates the agent cannot distinguish between partial and full task completion

### Why Finding Suitable Rewards is Hard

1. **Deceptive Local Optima**
   - Easy rewards (approaching object) can trap the agent
   - Agent learns to maximize easy rewards instead of completing the task
   - Example: Our agent achieved 100 reward/episode by repeatedly approaching and attempting grasps without success

2. **Competing Objectives**
   - Need to guide exploration (dense signal)
   - Need to prevent reward hacking (sparse signal)
   - These objectives often conflict

3. **Physical Constraints**
   - Grasping requires precise positioning and force control
   - Binary success/failure makes gradient-based learning difficult
   - Contact dynamics are discontinuous and hard to model

4. **Computational Cost**
   - Each reward design requires hours of training to evaluate
   - Bad reward designs waste significant compute time
   - Hyperparameter tuning compounds this issue

## Methodology

### Reward System Design Space

We identified three primary reward system architectures:

#### 1. Progressive Sub-Goal Rewards (Current Implementation)
```python
- Approach object: +1 point (distance < 10cm)
- Grasp attempt: +2 points (gripper closes near object)
- Successful grasp: +3 points (constraint created)
- Transport progress: +0-3 points (based on distance to goal)
- Task completion: +10 points total
```
**Issue**: Vulnerable to reward hacking - agent learns to collect approach+attempt rewards repeatedly

#### 2. Penalty-Based Dense Rewards
```python
- Failed grasp attempts: -0.5 penalty
- Distance-based exploration bonus: +0.1 max
- Successful grasp + transport: +5.0 * progress
- Task completion: +10 points
```
**Rationale**: Penalties discourage repeated failed attempts while maintaining exploration signal

#### 3. Staged/Hierarchical Rewards
```python
- No reward until successful grasp
- After grasp: +5.0 * transport_progress
- Task completion: +10 points
```
**Rationale**: Forces complete skill acquisition but may be too sparse for initial learning

### Evaluation Methodology

1. **Reward System Testing Protocol**
   - Quick evaluation script (`test_reward_fix.py`) to analyze reward distributions
   - Metrics: mean reward, grasp attempts/episode, actual grasp success rate
   - Compare reward accumulation patterns across schemes

2. **Training Evaluation Metrics**
   - Success rate (primary metric)
   - Episode reward mean/std
   - Value function loss (indicator of learning stability)
   - Policy loss convergence
   - Time to first success

3. **Decision Criteria**
   - Success rate > 50% within 500k timesteps
   - Value loss < 50 (indicates good value predictions)
   - Monotonic improvement in success rate
   - No reward hacking behavior

## Log

### Timeline of Attempts

#### Phase 1: Initial Hyperparameter Search
- **Approach**: Tune PPO hyperparameters with existing reward system
- **Result**: Best config achieved only 1% success rate
- **Learning**: Hyperparameters alone cannot fix fundamental reward issues
- **Key Config Found**:
  ```
  learning_rate: 2.93e-03
  n_steps: 4096
  batch_size: 512
  ```

#### Phase 2: Problem Diagnosis
- **Discovery**: Agent collecting ~100 reward/episode without completing task
- **Analysis**: Reward hacking - agent learns to approach + attempt grasp 33 times/episode
- **Evidence**: 
  - High value loss (132.56)
  - Stable but suboptimal policy
  - Consistent partial rewards without task completion

#### Phase 3: Reward System Redesign (Current Stage)
- **Status**: Testing alternative reward schemes
- **Next Steps**:
  1. Run `test_reward_fix.py` to evaluate reward distributions
  2. Implement most promising reward system
  3. Train from scratch (no checkpoint due to corrupted value function)
  4. Monitor for improved success rate

### Future Paths if Current Approach Fails

1. **Curriculum Learning**
   - Start with block closer to target
   - Gradually increase initial distance
   - May help with sparse reward schemes

2. **Reward Shaping with Potential Functions**
   - Use potential-based reward shaping to maintain optimal policy
   - Design potential function based on task progress

3. **Hybrid IL+RL Approach**
   - Generate small expert dataset
   - Use DAPG with sparse rewards
   - Leverages both demonstration and exploration

4. **Action Space Modification**
   - Discrete action space for key decisions (grasp/release)
   - May simplify credit assignment

5. **Two-Stage Training**
   - Stage 1: Learn grasping with simplified reward
   - Stage 2: Learn transport with frozen grasping policy

### Key Insights Learned

1. **Reward hacking is the primary failure mode** - not hyperparameters or exploration
2. **Value function quality is crucial** - high value loss indicates fundamental issues
3. **Sub-goal rewards must be carefully balanced** - too generous leads to local optima
4. **Testing reward schemes before full training saves time** - use simplified evaluation scripts

### Current Hypothesis

The penalty-based reward system is most promising because it:
- Maintains exploration signal through distance bonus
- Actively discourages failed grasp attempts
- Only provides significant rewards for actual progress
- Should prevent the reward hacking behavior observed in current system