# Demonstration Guide: Training and Evaluation Commands

This guide provides a complete reference for all training and evaluation commands across different algorithms and environments in the dVRK simulation project.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Expert Data Generation](#expert-data-generation)
3. [Behavioral Cloning (BC)](#behavioral-cloning-bc)
4. [Pure Reinforcement Learning (PPO)](#pure-reinforcement-learning-ppo)
5. [PPO with Imitation Learning (DAPG/PPO+IL)](#ppo-with-imitation-learning-dapgppoil)
6. [Evaluation Commands](#evaluation-commands)

## Environment Setup

### Start Docker Environment
```bash
# Start the development container
docker compose -f docker/docker-compose.yml up -d

# Verify environment setup
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 -c "import gymnasium as gym; import dvrk_gym; env = gym.make('PegTransfer-v0'); print('Environment test passed')"
```

### Stop Docker Environment
```bash
docker compose -f docker/docker-compose.yml down
```

## Expert Data Generation

### NeedleReach Task
```bash
# Generate expert demonstrations for NeedleReach
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/generate_expert_data_needle_reach.py

# Output: data/expert_data_needle_reach.pkl
```

### PegTransfer Task
```bash
# Generate expert demonstrations for PegTransfer
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/generate_expert_data_peg_transfer.py

# Output: data/expert_data_peg_transfer.pkl
```

## Behavioral Cloning (BC)

### Training BC Models

#### NeedleReach
```bash
# Default training (uses default expert data)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py --env NeedleReach-v0

# Custom hyperparameters
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py \
    --env NeedleReach-v0 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3

# With custom expert data
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py \
    --env NeedleReach-v0 \
    --expert-data data/my_expert_data.pkl
```

#### PegTransfer
```bash
# Default training
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py --env PegTransfer-v0

# Extended training for better performance
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py \
    --env PegTransfer-v0 \
    --epochs 150 \
    --batch-size 128
```

### Evaluating BC Models

#### NeedleReach
```bash
# Standard evaluation (100 episodes)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
    --model-path models/bc_needle_reach_XXXXXXXXXX.zip

# Quick evaluation (10 episodes)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
    --model-path models/bc_needle_reach_XXXXXXXXXX.zip \
    --episodes 10

# Without rendering (faster)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
    --model-path models/bc_needle_reach_XXXXXXXXXX.zip \
    --no-render
```

#### PegTransfer
```bash
# Standard evaluation
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
    --model-path models/bc_peg_transfer_XXXXXXXXXX.zip \
    --env PegTransfer-v0

# Extended evaluation (50 episodes)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
    --model-path models/bc_peg_transfer_XXXXXXXXXX.zip \
    --env PegTransfer-v0 \
    --episodes 50
```

## Pure Reinforcement Learning (PPO)

### Training PPO Models (Dense Rewards)

#### NeedleReach
```bash
# Default optimized training (100k steps, lr=3e-4)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py

# Extended training
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py \
    --timesteps 200000

# Custom hyperparameters
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py \
    --timesteps 150000 \
    --learning-rate 5e-4 \
    --n-steps 2048 \
    --batch-size 64
```

### Evaluating PPO Models

#### NeedleReach
```bash
# IMPORTANT: Must use --dense-reward flag for PPO models
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/ppo_needle_reach_XXXXXXXXXX.zip \
    --env-name NeedleReach-v0 \
    --flatten \
    --dense-reward

# Extended evaluation
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/ppo_needle_reach_XXXXXXXXXX.zip \
    --env-name NeedleReach-v0 \
    --n-episodes 50 \
    --flatten \
    --dense-reward
```

## PPO with Imitation Learning (DAPG/PPO+IL)

### Training DAPG Models (Sparse Rewards)

#### NeedleReach
```bash
# Auto-detect BC model and use default settings (300k steps, bc_weight=0.05)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py

# Specify BC model explicitly
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --bc-model models/bc_needle_reach_XXXXXXXXXX.zip

# Custom hyperparameters
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --timesteps 400000 \
    --bc-weight 0.1

# Without BC initialization (pure PPO+IL)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --bc-model none
```

#### PegTransfer
```bash
# Auto-detect BC model and use default settings (500k steps, bc_weight=0.02)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py --env PegTransfer-v0

# Higher BC weight for stronger guidance
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --env PegTransfer-v0 \
    --bc-weight 0.1

# Extended training
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --env PegTransfer-v0 \
    --timesteps 1000000 \
    --bc-weight 0.05

# Custom expert data
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --env PegTransfer-v0 \
    --expert-data data/expert_data_peg_transfer_v2.pkl
```

### Evaluating DAPG Models

#### NeedleReach
```bash
# IMPORTANT: Do NOT use --dense-reward for DAPG models (they train with sparse rewards)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/dapg_needle_reach_XXXXXXXXXX.zip \
    --env-name NeedleReach-v0 \
    --flatten

# Extended evaluation
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/dapg_needle_reach_XXXXXXXXXX.zip \
    --env-name NeedleReach-v0 \
    --n-episodes 100 \
    --flatten
```

#### PegTransfer
```bash
# Standard evaluation
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/dapg_peg_transfer_XXXXXXXXXX.zip \
    --env-name PegTransfer-v0 \
    --flatten

# Detailed evaluation (50 episodes)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/dapg_peg_transfer_XXXXXXXXXX.zip \
    --env-name PegTransfer-v0 \
    --n-episodes 50 \
    --flatten
```

## Evaluation Commands

### Quick Model Comparison
```bash
# Compare all models on NeedleReach
# BC Model
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
    --model-path models/bc_needle_reach_latest.zip \
    --episodes 20

# PPO Model (with dense reward)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/ppo_needle_reach_latest.zip \
    --env-name NeedleReach-v0 \
    --n-episodes 20 \
    --flatten \
    --dense-reward

# DAPG Model (sparse reward)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path models/dapg_needle_reach_latest.zip \
    --env-name NeedleReach-v0 \
    --n-episodes 20 \
    --flatten
```

### Batch Evaluation Script
```bash
# Create a batch evaluation script
cat > /tmp/batch_eval.sh << 'EOF'
#!/bin/bash
echo "Evaluating all models..."

# BC Models
for model in models/bc_*.zip; do
    echo "Evaluating BC model: $model"
    # Determine environment from model name
    if [[ "$model" == *"peg_transfer"* ]]; then
        ENV_NAME="PegTransfer-v0"
    else
        ENV_NAME="NeedleReach-v0"
    fi
    docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py \
        --model-path "$model" \
        --env "$ENV_NAME" \
        --episodes 20 \
        --no-render
done

# PPO Models (NeedleReach only)
for model in models/ppo_needle_reach_*.zip; do
    echo "Evaluating PPO model: $model"
    docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
        --model-path "$model" \
        --env-name NeedleReach-v0 \
        --n-episodes 20 \
        --flatten \
        --dense-reward
done

# DAPG Models
for model in models/dapg_*.zip; do
    echo "Evaluating DAPG model: $model"
    docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
        --model-path "$model" \
        --env-name "${model#*dapg_}" \
        --n-episodes 20 \
        --flatten
done
EOF

chmod +x /tmp/batch_eval.sh
```

## Advanced Usage

### Custom Training Pipeline
```bash
# 1. Generate expert data with specific parameters
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/generate_expert_data_peg_transfer.py

# 2. Train BC baseline
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py \
    --env PegTransfer-v0 \
    --epochs 100

# 3. Use BC model to initialize DAPG
BC_MODEL=$(docker compose -f docker/docker-compose.yml exec dvrk-dev ls -t models/bc_peg_transfer_*.zip | head -1)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
    --env PegTransfer-v0 \
    --bc-model $BC_MODEL \
    --timesteps 750000 \
    --bc-weight 0.08

# 4. Evaluate final performance
DAPG_MODEL=$(docker compose -f docker/docker-compose.yml exec dvrk-dev ls -t models/dapg_peg_transfer_*.zip | head -1)
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py \
    --model-path $DAPG_MODEL \
    --env-name PegTransfer-v0 \
    --n-episodes 100 \
    --flatten
```

### Hyperparameter Sweep
```bash
# BC weight sweep for DAPG
for bc_weight in 0.01 0.05 0.1 0.15; do
    echo "Training with bc_weight=$bc_weight"
    docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py \
        --env PegTransfer-v0 \
        --bc-weight $bc_weight \
        --timesteps 300000
done
```

## Notes

1. **Model Naming Convention**: Models are saved with timestamps (e.g., `bc_peg_transfer_1234567890.zip`)
2. **Environment Specification**: When evaluating ANY model trained on PegTransfer, you MUST include `--env PegTransfer-v0` in the command. The default environment is NeedleReach-v0, which has different observation dimensions and will cause errors.
3. **Flattening Requirement**: RL models require `--flatten` flag during evaluation due to dict observations
4. **Reward Types**: 
   - PPO uses dense rewards for exploration
   - DAPG uses sparse rewards to avoid reward hacking
   - BC is reward-agnostic
5. **Docker Requirement**: All commands must be run through Docker for reproducibility
6. **PegTransfer Note**: Pure PPO training is not recommended for PegTransfer due to task complexity. Use BC or DAPG instead.