# Manual PPO Curriculum Learning for PegTransfer Task

This simplified version allows you to manually train each curriculum level independently, giving you full control over the training process.

## Quick Start - Manual Training

### 1. Train Each Level Manually

```bash
# Train Level 1 (Precise Approach)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 1 --run-tag level1_baseline

# Train Level 2 (Precise Grasp) - starting from Level 1 model
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 2 --model-path models/ppo_curriculum/runs/run_20250729_143022_level1_baseline/model_level_1_final.zip --run-tag level2_from_level1

# Train Level 3 (Precise Transport) - starting from Level 2 model
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 3 --model-path models/ppo_curriculum/runs/run_20250729_150000_level2_from_level1/model_level_2_final.zip --run-tag level3_from_level2

# Train Level 4 (Full Task) - starting from Level 3 model
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 4 --model-path models/ppo_curriculum/runs/run_20250729_160000_level3_from_level2/model_level_3_final.zip --run-tag level4_final
```

### 2. Key Arguments

- `--level`: **Required.** Which curriculum level to train (1-4)
- `--model-path`: Path to a saved model to continue training from (optional)
- `--timesteps`: How many timesteps to train (uses config default if not specified)
- `--run-tag`: Tag to add to the auto-generated run name
- `--run-name`: Custom run name (instead of auto-generated)
- `--reset-timesteps`: Reset the timestep counter when loading a model

### 3. Directory Structure

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

### 4. Monitor Training

```bash
# View TensorBoard for specific run
docker compose -f docker/docker-compose.yml exec dvrk-dev tensorboard --logdir logs/ppo_curriculum/runs/run_20250729_143022_level1_baseline/tensorboard

# View all runs
docker compose -f docker/docker-compose.yml exec dvrk-dev tensorboard --logdir logs/ppo_curriculum/runs
```

### 5. Evaluate Models

```bash
# Evaluate a specific run (will find models for all levels in that run)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py --run-name run_20250729_143022_level1_baseline

# Evaluate specific model file
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py --model-path models/ppo_curriculum/runs/run_20250729_143022_level1_baseline/model_level_1_final.zip --level 1
```

## Advantages of Manual Training

1. **Full Control**: You decide when to advance levels based on performance
2. **Flexibility**: Train levels for different amounts of time
3. **Experimentation**: Easy to try different hyperparameters per level
4. **Debugging**: Simpler to debug issues with specific levels
5. **Resume Training**: Continue training a level if performance isn't good enough

## Curriculum Levels

| Level | Name | Description | Default Timesteps |
|-------|------|-------------|------------------|
| 1 | Precise Approach | Master reaching object with stable positioning | 50,000 |
| 2 | Precise Grasp | Master grasping after successful approach | 100,000 |
| 3 | Precise Transport | Master transporting without dropping | 150,000 |
| 4 | Full Task Mastery | Complete the full task with release precision | 200,000 |

## Example Training Workflow

```bash
# 1. Start with Level 1
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 1 --timesteps 100000 --run-tag experiment1

# 2. Check the success rate in the output. If good (>80%), move to Level 2
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/train_ppo_curriculum.py --level 2 --model-path models/ppo_curriculum/runs/run_XXXXXXXX_experiment1/model_level_1_final.zip --timesteps 150000 --run-tag experiment1

# 3. Continue this pattern through all levels
# 4. Compare different runs
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/compare_curriculum_runs.py

# 5. Test final model
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py --run-name run_XXXXXXXX_experiment1 --level 4 --render
```

## Training Tips

1. **Monitor Success Rates**: The training will print progress every 100 episodes
2. **Use Different Timesteps**: Some levels may need more training than others
3. **Save Interrupted Models**: If you interrupt training with Ctrl+C, the model is still saved
4. **Experiment with Hyperparameters**: Edit `curriculum_config.py` to try different settings
5. **Use Meaningful Tags**: Use descriptive run tags to organize experiments

## Configuration

You can still modify training parameters in `curriculum_config.py`:
- `total_timesteps_per_level`: Default timesteps for each level
- PPO hyperparameters for each level
- Episode length limits and early exit conditions

The main difference is that level advancement is now manual - you decide when to move to the next level based on the training results.