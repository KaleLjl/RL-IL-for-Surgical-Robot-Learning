# 7-Level Waypoint-Aligned Curriculum Learning for PegTransfer Task

This curriculum learning system provides perfect 1:1 mapping with oracle waypoints, decomposing the complete manipulation task into 7 focused skills that eliminate exploration issues and ensure smooth learning progression.

## Quick Start - Complete 7-Level Training

### 1. Train Each Level Sequentially

```bash
# Level 1: Waypoint 0 - Safe collision-free approach above object
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 1

# Level 2: Waypoint 1 - Precise positioning for grasp (approach from above)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 2 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_1_final.zip

# Level 3: Waypoint 2 - Grasping action execution
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 3 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_2_final.zip

# Level 4: Waypoint 3 - Coordinated lifting while maintaining grasp
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 4 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_3_final.zip

# Level 5: Waypoint 4 - Horizontal transport to above goal position
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 5 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_4_final.zip

# Level 6: Waypoint 5 - Precise lowering to release height
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 6 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_5_final.zip

# Level 7: Waypoint 6 - Release timing and task completion
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 7 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_6_final.zip
```

### 2. Key Arguments

- `--level`: **Required.** Which curriculum level to train (1-7)
- `--model-path`: Path to a saved model to continue training from (optional)
- `--timesteps`: How many timesteps to train (uses config default if not specified)
- `--render`: Enable visualization during training (opens PyBullet window)
- `--env`: Environment name (default: PegTransfer-v0)

### 3. Continue Training from Saved Model

To continue training from a previously saved model:

```bash
# Continue training from a specific model checkpoint
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level <level> --model-path <path_to_saved_model.zip>

# Example: Continue training Level 3 from a saved model
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 3 --model-path models/ppo_curriculum/runs/run_20250729_143022/model_level_3_final.zip

# Continue with custom timesteps
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 3 --model-path models/ppo_curriculum/runs/run_20250729_143022/model_level_3_final.zip --timesteps 50000
```

### 4. Directory Structure

Each training session creates its own run directory:

```
models/ppo_curriculum/runs/
├── run_20250729_143022_level1_baseline/
│   ├── metadata.json
│   ├── model_level_1_final.zip
│   └── checkpoints/
└── run_20250729_150000_level2_from_level1/
    ├── metadata.json
    ├── model_level_2_final.zip
    └── checkpoints/

logs/ppo_curriculum/runs/
├── run_20250729_143022_level1_baseline/
│   ├── training_summary_level_1.json
│   └── tensorboard/
└── run_20250729_150000_level2_from_level1/
    ├── training_summary_level_2.json
    └── tensorboard/
```

### 5. Monitor Training

```bash
# View TensorBoard for specific run
docker compose -f docker/docker-compose.yml exec dvrk-dev tensorboard --logdir logs/ppo_curriculum/runs/run_20250729_143022_level1_baseline/tensorboard

# View all runs
docker compose -f docker/docker-compose.yml exec dvrk-dev tensorboard --logdir logs/ppo_curriculum/runs
```

### 6. Evaluate Models

```bash
# Evaluate a specific model (level auto-detected from filename)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/run_20250729_143022_level1_baseline/model_level_1_final.zip

# Evaluate with visualization (10 episodes for quick testing)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/run_20250729_143022_level1_baseline/model_level_1_final.zip --render --episodes 10

# Override level detection manually
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/run_20250729_143022_level1_baseline/model_level_1_final.zip --level 2
```

#### Evaluation Options (Super Simple)

- `model_path`: **Required.** Path to the model file
- `--level`: Override auto-detected level (1-7)
- `--episodes`: Number of test episodes (default: 50)
- `--render`: Show visualization during evaluation

The script will:
1. Auto-detect the curriculum level from the filename
2. Test the model on that level
3. Print success rate and average reward
4. Save results as JSON file next to the model

## Advantages of Manual Training

1. **Full Control**: You decide when to advance levels based on performance
2. **Flexibility**: Train levels for different amounts of time
3. **Experimentation**: Easy to try different hyperparameters per level
4. **Debugging**: Simpler to debug issues with specific levels
5. **Resume Training**: Continue training a level if performance isn't good enough

## Complete 7-Level Waypoint-Aligned Curriculum

| Level | Oracle Waypoint | Name | Description | Default Timesteps |
|-------|----------------|------|-------------|------------------|
| **1** | **Waypoint 0** | Safe Approach | Collision-free approach above object (gripper open) | 100,000 |
| **2** | **Waypoint 1** | Precise Position | Precise positioning for grasp (gripper open) | 100,000 |
| **3** | **Waypoint 2** | Grasp Action | Execute grasping (close gripper) | 100,000 |
| **4** | **Waypoint 3** | Coordinated Lift | Lift while maintaining grasp | 120,000 |
| **5** | **Waypoint 4** | Transport | Horizontal transport to goal area | 150,000 |
| **6** | **Waypoint 5** | Precise Lower | Lower to release height | 100,000 |
| **7** | **Waypoint 6** | Release & Complete | Release object and complete task | 120,000 |

### Key Benefits of This Design:

1. **Zero Exploration Issues**: Each level has a clear, collision-free goal
2. **Perfect Skill Transfer**: Level N's success state = Level N+1's starting state  
3. **Oracle Alignment**: Every level teaches exactly one oracle waypoint
4. **Gradual Complexity**: Skills build incrementally without gaps
5. **Robust Learning**: No distribution shift between levels

## Example Training Workflow

### Complete 7-Level Training Sequence

```bash
# Step 1: Train Level 1 (Safe Approach) - Should achieve ~100% success
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 1

# Step 2: Use Level 1 model to train Level 2 (Precise Positioning)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 2 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_1_final.zip

# Step 3: Continue through all 7 levels...
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 3 --model-path models/ppo_curriculum/runs/run_TIMESTAMP/model_level_2_final.zip

# ... (repeat for levels 4-7)

# Final Step: Evaluate complete task performance
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/run_TIMESTAMP/model_level_7_final.zip --render --episodes 20
```

### Training with Visualization (Debugging)

```bash
# Train any level with visualization to debug issues
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/pgo_peg_transfer_curriculum/train_ppo_curriculum.py --level 1 --render

# Evaluate with visualization to see learned behavior
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/run_TIMESTAMP/model_level_1_final.zip --render --episodes 10
```

## Training Tips

1. **Monitor Success Rates**: The training will print progress every 50 episodes
2. **Use Different Timesteps**: Some levels may need more training than others
3. **Save Interrupted Models**: If you interrupt training with Ctrl+C, the model is still saved
4. **Experiment with Hyperparameters**: Edit `curriculum_config.py` to try different settings
5. **Visual Debugging**: Use `--render` flag to watch training in real-time (slower but helpful for debugging)

## Configuration

You can still modify training parameters in `curriculum_config.py`:
- `total_timesteps_per_level`: Default timesteps for each level
- PPO hyperparameters for each level
- Episode length limits and early exit conditions

The main difference is that level advancement is now manual - you decide when to move to the next level based on the training results.