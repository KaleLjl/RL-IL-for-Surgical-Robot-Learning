# Unified Training and Evaluation Infrastructure

This directory contains the unified infrastructure for training and evaluating all algorithms (BC, PPO, PPO+BC) on surgical manipulation tasks. The infrastructure is designed for systematic experimentation, hyperparameter tuning, and reproducible results generation.

## 🏗️ Architecture Overview

```
experiments/
├── train_unified.py         # Main training script for all algorithms
├── evaluate_unified.py      # Unified evaluation script
├── run_experiments.py       # Batch experiment runner
├── configs/                 # Configuration files
│   ├── base/               # Base configurations for each algorithm
│   └── experiments/        # Experiment-specific configs
├── demonstrations/          # Expert demonstration data
│   ├── expert_demo_needle_reach.pkl
│   └── expert_demo_peg_transfer.pkl
├── utils/                  # Utility modules
│   ├── config_parser.py    # Configuration management
│   └── logger.py           # Logging utilities
└── archive/                # Archived old scripts
```

## 🚀 Quick Start

### 1. Single Training Run

Train a BC model on Needle Reach:
```bash
python3 train_unified.py \
  --config configs/experiments/standard/bc_needle_reach.yaml \
  --task needle_reach \
  --demo-dir demonstrations
```

Train with custom parameters:
```bash
python3 train_unified.py \
  --config configs/base/bc_base.yaml \
  --task peg_transfer \
  --algorithm bc \
  --demo-dir demonstrations \
  --override training.peg_transfer.epochs=50 training.peg_transfer.learning_rate=1e-3
```

### 2. Evaluation

Evaluate a trained model:
```bash
python3 evaluate_unified.py \
  --model results/experiments/[experiment_name]/models/bc_final.zip \
  --task needle_reach \
  --algorithm bc \
  --n-episodes 100
```

### 3. Batch Experiments

Run multiple experiments from a batch config:
```bash
python3 run_experiments.py configs/experiments/sensitivity/bc_needle_reach_lr_sweep.yaml
```

## 📋 Supported Algorithms

| Algorithm | Status | Description |
|-----------|--------|-------------|
| `bc` | ✅ Ready | Behavioral Cloning with imitation learning |
| `ppo` | ✅ Ready | Proximal Policy Optimization |
| `ppo_bc` | 🚧 Planned | PPO with BC initialization/guidance |

## 📁 Configuration System

### Base Configurations

Located in `configs/base/`, these define default parameters:

- `bc_base.yaml` - Behavioral Cloning defaults
- `ppo_base.yaml` - PPO defaults  
- `ppo_bc_base.yaml` - PPO+BC defaults

### Experiment Configurations

Located in `configs/experiments/`:

- `standard/` - Best hyperparameters for each task
- `sensitivity/` - Hyperparameter sensitivity analysis
- `ablation/` - Component ablation studies
- `sample_efficiency/` - Varying demonstration counts

### Configuration Structure

```yaml
algorithm: "bc"
seed: 42

network:
  needle_reach:
    hidden_sizes: [256, 256]
    activation: "relu"
  peg_transfer:
    hidden_sizes: [128, 128]
    activation: "relu"

training:
  needle_reach:
    epochs: 200
    batch_size: 64
    learning_rate: 1.0e-4
    weight_decay: 1.0e-4
    
data:
  num_demonstrations: 100
  
logging:
  tensorboard: true
  csv_logging: true
  save_freq: 10
```

## 🎯 Command Line Interface

### Training Script (`train_unified.py`)

**Required Arguments:**
- `--config` - Path to configuration file
- `--task` - Task name (`needle_reach` or `peg_transfer`)

**Optional Arguments:**
- `--algorithm` - Algorithm override (`bc`, `ppo`, `ppo_bc`)
- `--demo-dir` - Directory containing expert demonstrations (default: `demonstrations`)
- `--output-dir` - Base directory for outputs (default: `results/experiments`)
- `--experiment-name` - Custom experiment name
- `--seed` - Random seed override
- `--override` - Config overrides in format `key=value` (can specify multiple)
- `--no-tensorboard` - Disable TensorBoard logging
- `--verbose` - Verbosity level (0-2)

**Examples:**
```bash
# Basic training
python3 train_unified.py --config configs/base/bc_base.yaml --task needle_reach

# With overrides
python3 train_unified.py --config configs/base/bc_base.yaml --task needle_reach \
  --override training.needle_reach.epochs=100 seed=123

# Custom experiment
python3 train_unified.py --config configs/base/ppo_base.yaml --task peg_transfer \
  --experiment-name ppo_peg_transfer_experiment --seed 42
```

### Evaluation Script (`evaluate_unified.py`)

**Required Arguments:**
- `--model` - Path to trained model
- `--task` - Task name
- `--algorithm` - Algorithm used for training

**Optional Arguments:**
- `--n-episodes` - Number of evaluation episodes (default: 100)
- `--config` - Path to original config file
- `--render` - Enable environment rendering
- `--output-dir` - Directory for evaluation outputs

**Examples:**
```bash
# Standard evaluation
python3 evaluate_unified.py \
  --model results/experiments/20240809_120000_bc_needle_reach_abc123/models/bc_final.zip \
  --task needle_reach \
  --algorithm bc

# Quick test with rendering
python3 evaluate_unified.py \
  --model path/to/model.zip \
  --task peg_transfer \
  --algorithm ppo \
  --n-episodes 5 \
  --render
```

### Batch Runner (`run_experiments.py`)

**Required Arguments:**
- `config` - Path to batch experiment configuration

**Optional Arguments:**
- `--base-dir` - Base directory for batch results (default: `results/experiments`)

**Example:**
```bash
python3 run_experiments.py configs/experiments/sensitivity/bc_lr_sweep.yaml
```

## 📊 Output Structure

### Single Experiment Output
```
results/experiments/[timestamp]_[algorithm]_[task]_[config_hash]/
├── config.json              # Complete configuration used
├── training.log              # Training progress log
├── metrics.csv               # Training metrics over time
├── models/                   # Saved model checkpoints
│   ├── [algo]_epoch_N.zip   # Periodic checkpoints
│   └── [algo]_final.zip     # Final trained model
├── evaluations/              # Evaluation results
│   └── evaluation_[algo]_[task].json
└── tensorboard/              # TensorBoard logs (if enabled)
```

### Batch Experiment Output
```
results/experiments/batch_[timestamp]/
├── experiment_config.yaml    # Original batch configuration
├── results.json              # Detailed results from all experiments
├── summary.csv               # Aggregated metrics table
├── [experiment_1]/           # Individual experiment directories
│   ├── config.yaml          # Experiment-specific config
│   └── [experiment_1]/      # Training output (same structure as above)
└── [experiment_2]/
    ├── config.yaml
    └── [experiment_2]/
```

## 🧪 Experiment Types

### 1. Standard Training

Best hyperparameters for each algorithm-task combination:

```bash
# BC on both tasks
python3 train_unified.py --config configs/experiments/standard/bc_needle_reach.yaml --task needle_reach
python3 train_unified.py --config configs/experiments/standard/bc_peg_transfer.yaml --task peg_transfer

# PPO on both tasks
python3 train_unified.py --config configs/experiments/standard/ppo_needle_reach.yaml --task needle_reach
python3 train_unified.py --config configs/experiments/standard/ppo_peg_transfer.yaml --task peg_transfer
```

### 2. Hyperparameter Sensitivity

Test robustness to hyperparameter choices:

```bash
# Learning rate sweep for BC
python3 run_experiments.py configs/experiments/sensitivity/bc_needle_reach_lr_sweep.yaml

# Network size sweep
python3 run_experiments.py configs/experiments/sensitivity/bc_peg_transfer_network_sweep.yaml

# BC loss weight sweep for PPO+BC
python3 run_experiments.py configs/experiments/sensitivity/ppo_bc_beta_sweep.yaml
```

### 3. Ablation Studies

Test importance of different components:

```bash
# BC initialization vs random initialization for PPO+BC
python3 run_experiments.py configs/experiments/ablation/ppo_bc_initialization.yaml

# Different regularization strengths
python3 run_experiments.py configs/experiments/ablation/bc_regularization.yaml
```

### 4. Sample Efficiency

Test performance with different numbers of demonstrations:

```bash
# BC with varying demo counts
python3 run_experiments.py configs/experiments/sample_efficiency/bc_demo_count.yaml
```

### 5. Multi-Seed Runs

For statistical significance:

```bash
# Run with multiple seeds
for seed in 42 123 456 789 999; do
  python3 train_unified.py \
    --config configs/experiments/standard/bc_needle_reach.yaml \
    --task needle_reach \
    --seed $seed \
    --experiment-name bc_needle_reach_seed_$seed
done
```

## 🔧 Creating Custom Experiments

### 1. Single Experiment Config

Create a new YAML file with your parameters:

```yaml
# configs/my_experiment.yaml
algorithm: "bc"
seed: 42

training:
  needle_reach:
    epochs: 150
    learning_rate: 5.0e-4
    batch_size: 32

network:
  needle_reach:
    hidden_sizes: [512, 512]

logging:
  save_freq: 25
```

### 2. Batch Experiment Config

Create a batch configuration for systematic testing:

```yaml
# configs/my_batch_experiment.yaml
task: needle_reach
algorithm: bc
base_config: "base/bc_base.yaml"

experiments:
  - name: "bc_small_net"
    network:
      needle_reach:
        hidden_sizes: [64, 64]
  
  - name: "bc_large_net"
    network:
      needle_reach:
        hidden_sizes: [512, 512]
  
  - name: "bc_high_lr"
    training:
      needle_reach:
        learning_rate: 1.0e-3
```

## 📈 Analyzing Results

### Training Metrics

Access training metrics via:

1. **TensorBoard**: `tensorboard --logdir results/experiments/[experiment]/tensorboard`
2. **CSV files**: `results/experiments/[experiment]/metrics.csv`
3. **Training logs**: `results/experiments/[experiment]/training.log`

### Evaluation Metrics

Evaluation results are saved as JSON:

```json
{
  "model_path": "path/to/model.zip",
  "task": "needle_reach",
  "algorithm": "bc",
  "n_episodes": 100,
  "metrics": {
    "success_rate": 0.85,
    "success_rate_ci": 0.07,
    "mean_reward": 42.3,
    "std_reward": 15.2,
    "mean_length": 45.7,
    "failure_reasons": {
      "timeout": 12,
      "failed_to_reach": 3
    }
  }
}
```

### Batch Results Analysis

Batch experiments generate:

1. **Summary CSV**: Quick comparison table
2. **Detailed JSON**: Complete results for each experiment
3. **Individual logs**: Full details for each experiment

Example analysis:

```python
import pandas as pd
import json

# Load batch results
summary = pd.read_csv('results/experiments/batch_*/summary.csv')
print(summary.describe())

# Load detailed results
with open('results/experiments/batch_*/results.json', 'r') as f:
    detailed = json.load(f)
```

## 🐳 Docker Usage

The infrastructure is designed to work seamlessly with Docker:

```bash
# Run training in Docker container
docker exec dvrk_dev bash -c "cd /app/experiments && python3 train_unified.py --config configs/base/bc_base.yaml --task needle_reach --demo-dir demonstrations --no-tensorboard"

# Run evaluation in Docker
docker exec dvrk_dev bash -c "cd /app/experiments && python3 evaluate_unified.py --model results/experiments/[experiment]/models/bc_final.zip --task needle_reach --algorithm bc"
```

## ⚠️ Important Notes

### Data Requirements

- Expert demonstrations must be available in `demonstrations/` directory
- Demonstration files should be named: `expert_demo_[task].pkl`
- Current format: List of dictionaries with `obs` and `acts` keys

**Generating Demonstrations:**
```bash
# First, generate expert demonstrations
python3 generate_expert_data_needle_reach.py
python3 generate_expert_data_peg_transfer.py
```

### Memory and Compute

- BC training: ~1-2 minutes for 200 epochs
- PPO training: ~2-3 hours for 100K steps on Needle Reach
- PPO training: ~6-8 hours for 300K steps on Peg Transfer
- Evaluation: ~1-2 minutes for 100 episodes

### File Paths

- All paths should be absolute or relative to the `experiments/` directory
- Models are automatically saved with timestamps and config hashes
- Use `--demo-dir demonstrations` when running from `experiments/` directory

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory and have all dependencies installed
2. **Demo File Not Found**: Check that demonstration files exist and path is correct
3. **Model Loading Fails**: Ensure you're using the correct algorithm parameter that matches the trained model
4. **Out of Memory**: Reduce batch size or use fewer parallel environments for PPO

### Debug Mode

Enable detailed logging:

```bash
python3 train_unified.py --config configs/base/bc_base.yaml --task needle_reach --verbose 2
```

### Log Files

Check the training logs for detailed error messages:

```bash
tail -f results/experiments/[experiment]/training.log
```

## 📚 Next Steps

1. **Run Standard Experiments**: Start with configs in `experiments/standard/`
2. **Hyperparameter Sweeps**: Use batch runner for systematic parameter exploration
3. **Multi-Seed Validation**: Run multiple seeds for statistical significance
4. **Result Analysis**: Use the generated CSV and JSON files for your thesis results
5. **Custom Experiments**: Create your own configs for specific research questions

For questions or issues, check the training logs and ensure all paths and dependencies are correctly set up.