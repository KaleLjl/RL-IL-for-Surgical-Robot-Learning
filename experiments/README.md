# Unified Training and Evaluation Infrastructure

This directory contains the unified infrastructure for training and evaluating all algorithms (BC, PPO, PPO+BC) on surgical manipulation tasks. The infrastructure is designed for systematic experimentation, hyperparameter tuning, and reproducible results generation.

## 🏗️ Architecture Overview

```
experiments/
├── train_unified.py         # Main training script for all algorithms
├── evaluate_unified.py      # Unified evaluation script
├── run_experiments.py       # Batch experiment runner
├── hyperopt_unified.py      # Hyperparameter optimization using Optuna
├── configs/                 # Configuration files
│   ├── base/               # Base configurations for each algorithm
│   └── experiments/        # Experiment-specific configs
│       ├── hyperparameter/ # Hyperparameter optimization configs
│       ├── standard/       # Best hyperparameters for each task
│       ├── sensitivity/    # Hyperparameter sensitivity analysis
│       ├── ablation/       # Component ablation studies
│       └── sample_efficiency/ # Varying demonstration counts
├── data/                    # Expert demonstration data
│   ├── expert_data_needle_reach.pkl
│   └── expert_data_peg_transfer.pkl
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
  --task needle_reach
```

Train with custom parameters:
```bash
python3 train_unified.py \
  --config configs/base/bc_base.yaml \
  --task peg_transfer \
  --algorithm bc \
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

### 4. Hyperparameter Optimization 🔬

**NEW**: Automated hyperparameter optimization using Optuna with Tree-structured Parzen Estimator (TPE):

#### Single Algorithm Optimization
```bash
# Optimize BC hyperparameters for needle_reach
python3 hyperopt_unified.py --algorithm bc --task needle_reach --n-trials 50

# Optimize PPO hyperparameters for needle_reach  
python3 hyperopt_unified.py --algorithm ppo --task needle_reach --n-trials 30

# Optimize PPO+BC (requires trained BC model)
python3 hyperopt_unified.py --algorithm ppo_bc --task needle_reach --n-trials 30 \
  --bc-model-path results/hyperopt/bc_needle_reach_*/models/bc_final.zip
```

#### Sequential Optimization (Recommended)
```bash
# Automatic BC → PPO+BC sequential optimization
python3 hyperopt_unified.py --stage sequential --task needle_reach \
  --bc-trials 50 --ppo-bc-trials 30

# For peg_transfer (more challenging task)
python3 hyperopt_unified.py --stage sequential --task peg_transfer \
  --bc-trials 50 --ppo-bc-trials 30
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

- `hyperparameter/` - Automated hyperparameter optimization configs
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
- Demonstrations are automatically loaded from `data/` directory
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

### Hyperparameter Optimizer (`hyperopt_unified.py`) 🔬

**NEW**: Automated hyperparameter optimization using Optuna with intelligent search algorithms.

**Required Arguments:**
- `--algorithm` - Algorithm to optimize (`bc`, `ppo`, `ppo_bc`)
- `--task` - Task name (`needle_reach`, `peg_transfer`)

**Optional Arguments:**
- `--stage` - Optimization stage (`bc`, `ppo_bc`, `sequential`) 
- `--n-trials` - Number of optimization trials (default: 50)
- `--bc-model-path` - Path to BC model for `ppo_bc` stage
- `--study-name` - Custom study name
- `--output-dir` - Output directory (default: `results/hyperopt`)

**Sequential Mode Arguments:**
- `--bc-trials` - Number of BC trials (default: 50)
- `--ppo-bc-trials` - Number of PPO+BC trials (default: 30)

**Examples:**
```bash
# Single algorithm optimization
python3 hyperopt_unified.py --algorithm bc --task needle_reach --n-trials 50

# Sequential optimization (recommended workflow)
python3 hyperopt_unified.py --stage sequential --task needle_reach \
  --bc-trials 50 --ppo-bc-trials 30

# PPO+BC with specific BC model
python3 hyperopt_unified.py --algorithm ppo_bc --task needle_reach \
  --bc-model-path results/hyperopt/bc_best/models/bc_final.zip \
  --n-trials 30
```

**Features:**
- **TPE (Tree-structured Parzen Estimator)**: Smart hyperparameter search
- **Early Pruning**: Stops poor trials early to save compute time
- **Multi-objective**: Optimizes success rate as primary metric
- **Visualization**: Automatic generation of optimization plots
- **Sequential Workflow**: BC optimization → PPO+BC optimization
- **Reproducible**: Fixed seeds and complete experiment tracking

**Search Spaces:**
- **BC**: Network size, learning rate, batch size, weight decay, epochs
- **PPO**: All PPO parameters + network architecture + training timesteps
- **PPO+BC**: PPO parameters + BC loss weight + BC update frequency

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

### Hyperparameter Optimization Output
```
results/hyperopt/[study_name]/
├── study.json                     # Complete Optuna study results
├── optimal_config_[algo]_[task].yaml  # Best hyperparameters found
├── trials.csv                     # All trials in tabular format
├── optimization_history.html      # Interactive optimization plot
├── parameter_importance.html      # Parameter importance analysis
├── parallel_coordinates.html      # Multi-dimensional visualization
└── [individual_trial_dirs]/       # Each trial's full training output

# Sequential optimization output
results/hyperopt/sequential_[task]_[timestamp]/
├── sequential_summary_[task]_[timestamp].json  # Complete sequential results
├── bc_optimization/               # BC optimization study
│   ├── study.json
│   ├── optimal_config_bc_[task].yaml
│   └── [visualizations]
└── ppo_bc_optimization/           # PPO+BC optimization study
    ├── study.json
    ├── optimal_config_ppo_bc_[task].yaml
    └── [visualizations]
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

### 2. Hyperparameter Sensitivity & Optimization

#### Automated Optimization (Recommended) 🔬
Use Optuna for intelligent hyperparameter search:

```bash
# Automated BC optimization
python3 hyperopt_unified.py --algorithm bc --task needle_reach --n-trials 50

# Sequential BC → PPO+BC optimization
python3 hyperopt_unified.py --stage sequential --task needle_reach \
  --bc-trials 50 --ppo-bc-trials 30
```

#### Manual Sensitivity Analysis
Test specific hyperparameter ranges:

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
docker exec dvrk_dev bash -c "cd /app/experiments && python3 train_unified.py --config configs/base/bc_base.yaml --task needle_reach --no-tensorboard"

# Run evaluation in Docker
docker exec dvrk_dev bash -c "cd /app/experiments && python3 evaluate_unified.py --model results/experiments/[experiment]/models/bc_final.zip --task needle_reach --algorithm bc"
```

## ⚠️ Important Notes

### Data Requirements

- Expert demonstrations must be available in `data/` directory
- Demonstration files should be named: `expert_data_[task].pkl`
- Current format: List of dictionaries with `obs` and `acts` keys

**Generating Demonstrations:**
```bash
# Generate expert demonstrations (saves to data/ directory)
python3 generate_expert_data_needle_reach.py
python3 generate_expert_data_peg_transfer.py

# Training scripts automatically load from data/expert_data_{task}.pkl
```

### Memory and Compute

- BC training: ~1-2 minutes for 200 epochs
- PPO training: ~2-3 hours for 100K steps on Needle Reach
- PPO training: ~6-8 hours for 300K steps on Peg Transfer
- Evaluation: ~1-2 minutes for 100 episodes

### File Paths

- All paths should be absolute or relative to the `experiments/` directory
- Models are automatically saved with timestamps and config hashes
- Expert demonstrations are automatically loaded from `experiments/data/` directory

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

## 📈 Step-by-Step Guide: Systematic Results Generation

This section provides a complete roadmap for generating comprehensive experimental results for your thesis Chapter 4.

### 🎯 Phase 1: Data Preparation

**Step 1: Generate Expert Demonstrations**
```bash
# Generate demonstrations for both tasks
python3 generate_expert_data_needle_reach.py
python3 generate_expert_data_peg_transfer.py

# Verify data files exist
ls -la data/expert_data_*.pkl
```

**Step 2: Validate Demonstrations**
```bash
# Quick test with minimal training to ensure data loads correctly
python3 train_unified.py --config configs/base/bc_base.yaml --task needle_reach \
  --override training.needle_reach.epochs=1
```

### 🧪 Phase 2: Standard Training Experiments

**Step 3: Create Standard Experiment Configs**

First, create optimized configs for each algorithm-task combination:

```bash
# Create BC standard configs
cat > configs/experiments/standard/bc_needle_reach_final.yaml << 'EOF'
algorithm: "bc"
seed: 42

network:
  needle_reach:
    hidden_sizes: [256, 256]
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
  save_freq: 20
EOF

cat > configs/experiments/standard/bc_peg_transfer_final.yaml << 'EOF'
algorithm: "bc"
seed: 42

network:
  peg_transfer:
    hidden_sizes: [128, 128]
    activation: "relu"

training:
  peg_transfer:
    epochs: 25
    batch_size: 64
    learning_rate: 5.0e-5
    weight_decay: 1.0e-3

data:
  num_demonstrations: 100

logging:
  tensorboard: true
  csv_logging: true
  save_freq: 5
EOF
```

**Step 4: Run Standard Training**
```bash
# BC on both tasks
python3 train_unified.py --config configs/experiments/standard/bc_needle_reach_final.yaml --task needle_reach
python3 train_unified.py --config configs/experiments/standard/bc_peg_transfer_final.yaml --task peg_transfer

# PPO on needle reach (known to work)
python3 train_unified.py --config configs/base/ppo_base.yaml --task needle_reach

# Document PPO failure on peg transfer
python3 train_unified.py --config configs/base/ppo_base.yaml --task peg_transfer --override training.peg_transfer.total_timesteps=50000
# Expected: Document that PPO fails to learn peg transfer within reasonable time
```

### 🔍 Phase 3: Hyperparameter Sensitivity Analysis

**⚠️ Important**: Complete hyperparameter tuning BEFORE ablation and sample efficiency studies to ensure optimal baseline parameters.

**Step 5: BC Network Size Sensitivity**
```bash
cat > configs/experiments/sensitivity/bc_network_sensitivity.yaml << 'EOF'
task: needle_reach
algorithm: bc
base_config: "base/bc_base.yaml"

experiments:
  - name: "bc_small_64x64"
    network:
      needle_reach:
        hidden_sizes: [64, 64]
  
  - name: "bc_medium_128x128"
    network:
      needle_reach:
        hidden_sizes: [128, 128]
  
  - name: "bc_standard_256x256"
    network:
      needle_reach:
        hidden_sizes: [256, 256]
  
  - name: "bc_large_512x512"
    network:
      needle_reach:
        hidden_sizes: [512, 512]
EOF

# Run network sensitivity analysis
python3 run_experiments.py configs/experiments/sensitivity/bc_network_sensitivity.yaml
```

**Step 6: BC Learning Rate Sensitivity**
```bash
cat > configs/experiments/sensitivity/bc_lr_sensitivity.yaml << 'EOF'
task: needle_reach
algorithm: bc
base_config: "base/bc_base.yaml"

experiments:
  - name: "bc_lr_1e-5"
    training:
      needle_reach:
        learning_rate: 1.0e-5
  
  - name: "bc_lr_5e-5"
    training:
      needle_reach:
        learning_rate: 5.0e-5
  
  - name: "bc_lr_1e-4"
    training:
      needle_reach:
        learning_rate: 1.0e-4
  
  - name: "bc_lr_5e-4"
    training:
      needle_reach:
        learning_rate: 5.0e-4
  
  - name: "bc_lr_1e-3"
    training:
      needle_reach:
        learning_rate: 1.0e-3
EOF

python3 run_experiments.py configs/experiments/sensitivity/bc_lr_sensitivity.yaml
```

**Step 7: PPO+BC Beta Sensitivity** *(Future Implementation)*
```bash
# Template for when PPO+BC is implemented
cat > configs/experiments/sensitivity/ppo_bc_beta_sensitivity.yaml << 'EOF'
task: needle_reach
algorithm: ppo_bc
base_config: "base/ppo_bc_base.yaml"

experiments:
  - name: "ppo_bc_beta_0.01"
    bc:
      bc_loss_weight:
        needle_reach: 0.01
  
  - name: "ppo_bc_beta_0.02"
    bc:
      bc_loss_weight:
        needle_reach: 0.02
  
  - name: "ppo_bc_beta_0.05"
    bc:
      bc_loss_weight:
        needle_reach: 0.05
  
  - name: "ppo_bc_beta_0.1"
    bc:
      bc_loss_weight:
        needle_reach: 0.1
EOF

# python3 run_experiments.py configs/experiments/sensitivity/ppo_bc_beta_sensitivity.yaml
```

### 🎯 Phase 4: Create Optimized Standard Configurations

**⚠️ Critical Step**: After completing sensitivity analysis, create optimized standard configs with the best hyperparameters found.

**Step 8: Analyze Sensitivity Results and Identify Optimal Parameters**
```bash
# Analyze batch results to find best hyperparameters
python3 << 'EOF'
import json
import pandas as pd
from pathlib import Path

def find_best_hyperparameters(batch_dir):
    """Find best hyperparameters from sensitivity analysis."""
    results_file = Path(batch_dir) / 'results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract success rates for each experiment
        results = []
        for exp in data:
            if 'evaluation' in exp and 'metrics' in exp['evaluation']:
                results.append({
                    'name': exp['name'], 
                    'success_rate': exp['evaluation']['metrics']['success_rate'],
                    'mean_reward': exp['evaluation']['metrics']['mean_reward']
                })
        
        df = pd.DataFrame(results)
        best_config = df.loc[df['success_rate'].idxmax()]
        
        print("="*50)
        print("BEST HYPERPARAMETERS FOUND:")
        print("="*50)
        print(f"Best Config: {best_config['name']}")
        print(f"Success Rate: {best_config['success_rate']*100:.1f}%")
        print(f"Mean Reward: {best_config['mean_reward']:.2f}")
        print("="*50)
        
        return best_config

# Usage:
# best = find_best_hyperparameters('results/experiments/batch_[timestamp]')
EOF
```

**Step 9: Create Optimized Standard Configurations**
```bash
# Based on sensitivity analysis results, create optimized configs
# Example: If bc_lr_5e-5 and bc_network_256x256 performed best

cat > configs/experiments/standard/bc_needle_reach_optimized.yaml << 'EOF'
algorithm: "bc"
seed: 42

network:
  needle_reach:
    hidden_sizes: [256, 256]  # Best from network sensitivity
    activation: "relu"

training:
  needle_reach:
    epochs: 200               # Sufficient for convergence
    batch_size: 64
    learning_rate: 5.0e-5     # Best from LR sensitivity
    weight_decay: 1.0e-4      # Best from regularization analysis
    early_stopping_patience: 20
    early_stopping_threshold: 0.001

data:
  num_demonstrations: 100

logging:
  tensorboard: true
  csv_logging: true
  save_freq: 20
  verbose: 1
EOF

cat > configs/experiments/standard/bc_peg_transfer_optimized.yaml << 'EOF' 
algorithm: "bc"
seed: 42

network:
  peg_transfer:
    hidden_sizes: [128, 128]  # Best from network sensitivity
    activation: "relu"

training:
  peg_transfer:
    epochs: 25                # Best from convergence analysis  
    batch_size: 64
    learning_rate: 5.0e-5     # Best from LR sensitivity
    weight_decay: 1.0e-3      # Higher regularization for complex task
    early_stopping_patience: 10
    early_stopping_threshold: 0.001

data:
  num_demonstrations: 100

logging:
  tensorboard: true
  csv_logging: true 
  save_freq: 5
  verbose: 1
EOF
```

**Step 10: Validate Optimized Configurations**
```bash
# Test optimized configs with multiple seeds to ensure robustness
for seed in 42 123 456; do
    python3 train_unified.py \
        --config configs/experiments/standard/bc_needle_reach_optimized.yaml \
        --task needle_reach \
        --seed $seed \
        --experiment-name "bc_needle_reach_optimized_seed_${seed}"
done

# Verify consistent high performance across seeds
```

### 🧬 Phase 5: Ablation Studies

**⚠️ Important**: Use optimized standard configurations as the baseline for all ablation studies to ensure fair comparisons.

**Step 11: PPO+BC Initialization Ablation** *(Future Implementation)*
```bash
cat > configs/experiments/ablation/ppo_bc_initialization.yaml << 'EOF'
task: needle_reach
algorithm: ppo_bc
base_config: "base/ppo_bc_base.yaml"

experiments:
  - name: "ppo_bc_random_init"
    bc:
      use_bc_initialization: false
      bc_model_path: null
  
  - name: "ppo_bc_pretrained_init"
    bc:
      use_bc_initialization: true
      bc_model_path: "results/experiments/[bc_model_path]/models/bc_final.zip"
EOF

# python3 run_experiments.py configs/experiments/ablation/ppo_bc_initialization.yaml
```

### 📊 Phase 6: Sample Efficiency Analysis  

**⚠️ Important**: Use optimized standard configurations as the baseline for all sample efficiency studies.

**Step 12: Varying Demonstration Counts for BC**
```bash
cat > configs/experiments/sample_efficiency/bc_demo_count.yaml << 'EOF'
task: needle_reach
algorithm: bc
base_config: "base/bc_base.yaml"

experiments:
  - name: "bc_demos_25"
    data:
      num_demonstrations: 25
    training:
      needle_reach:
        epochs: 100  # Fewer epochs for less data
  
  - name: "bc_demos_50"
    data:
      num_demonstrations: 50
    training:
      needle_reach:
        epochs: 150
  
  - name: "bc_demos_100"
    data:
      num_demonstrations: 100
    training:
      needle_reach:
        epochs: 200
  
  - name: "bc_demos_150"
    data:
      num_demonstrations: 150
    training:
      needle_reach:
        epochs: 250

# Note: You'll need to modify the data loading to support subset sampling
EOF

# First implement subset sampling in train_unified.py, then run:
# python3 run_experiments.py configs/experiments/sample_efficiency/bc_demo_count.yaml
```

### 🎲 Phase 7: Multi-Seed Statistical Validation

**Step 13: Multi-Seed Runs for Statistical Significance**
```bash
# Create script for multi-seed runs
cat > run_multiseed.sh << 'EOF'
#!/bin/bash

SEEDS=(42 123 456 789 999)
CONFIG=$1
TASK=$2
BASE_NAME=$3

for seed in "${SEEDS[@]}"; do
    echo "Running seed $seed..."
    python3 train_unified.py \
        --config "$CONFIG" \
        --task "$TASK" \
        --seed $seed \
        --experiment-name "${BASE_NAME}_seed_${seed}"
done
EOF

chmod +x run_multiseed.sh

# Run multi-seed experiments for key results using OPTIMIZED configs
./run_multiseed.sh configs/experiments/standard/bc_needle_reach_optimized.yaml needle_reach bc_needle_reach_multiseed
./run_multiseed.sh configs/experiments/standard/bc_peg_transfer_optimized.yaml peg_transfer bc_peg_transfer_multiseed
```

### 📈 Phase 8: Results Analysis and Visualization

**Step 14: Extract Training Metrics**
```bash
# Create analysis script
cat > analyze_results.py << 'EOF'
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_batch_results(batch_dir):
    """Analyze results from batch experiments."""
    results_file = Path(batch_dir) / 'results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract success rates
        success_rates = []
        for exp in data:
            if 'evaluation' in exp and 'metrics' in exp['evaluation']:
                success_rates.append({
                    'name': exp['name'],
                    'success_rate': exp['evaluation']['metrics']['success_rate'],
                    'mean_reward': exp['evaluation']['metrics']['mean_reward']
                })
        
        return pd.DataFrame(success_rates)
    
    return None

# Usage:
# df = analyze_batch_results('results/experiments/batch_[timestamp]')
# print(df.describe())
EOF
```

**Step 15: Generate Thesis Figures and Tables**
```bash
# Create comprehensive results summary
cat > generate_thesis_results.py << 'EOF'
import json
import pandas as pd
import numpy as np
from pathlib import Path

def generate_results_table():
    """Generate LaTeX table for thesis."""
    
    # Collect all evaluation results
    results = []
    
    # Scan all experiment directories
    for exp_dir in Path('results/experiments').glob('*'):
        if exp_dir.is_dir() and 'batch_' not in exp_dir.name:
            eval_file = exp_dir / 'evaluations' / 'evaluation_*.json'
            eval_files = list(exp_dir.glob('evaluations/evaluation_*.json'))
            
            if eval_files:
                with open(eval_files[0], 'r') as f:
                    data = json.load(f)
                
                results.append({
                    'algorithm': data['algorithm'],
                    'task': data['task'],
                    'success_rate': data['metrics']['success_rate'] * 100,
                    'success_rate_ci': data['metrics'].get('success_rate_ci', 0) * 100,
                    'mean_reward': data['metrics']['mean_reward'],
                    'mean_length': data['metrics']['mean_length']
                })
    
    df = pd.DataFrame(results)
    
    # Generate LaTeX table
    latex_table = df.groupby(['algorithm', 'task']).agg({
        'success_rate': ['mean', 'std'],
        'mean_reward': ['mean', 'std'],
        'mean_length': ['mean', 'std']
    }).round(2)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|l|c|c|c|}")
    print("\\hline")
    print("Algorithm & Task & Success Rate (\\%) & Mean Reward & Mean Length \\\\")
    print("\\hline")
    
    for (algo, task), row in latex_table.iterrows():
        success_mean = row[('success_rate', 'mean')]
        success_std = row[('success_rate', 'std')]
        reward_mean = row[('mean_reward', 'mean')]
        length_mean = row[('mean_length', 'mean')]
        
        print(f"{algo.upper()} & {task.replace('_', ' ').title()} & "
              f"{success_mean:.1f} $\\pm$ {success_std:.1f} & "
              f"{reward_mean:.1f} & {length_mean:.1f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Experimental Results Summary}")
    print("\\label{tab:results}")
    print("\\end{table}")

if __name__ == "__main__":
    generate_results_table()
EOF

python3 generate_thesis_results.py > thesis_results_table.tex
```

### ✅ Phase 9: Final Verification

**Step 16: Results Checklist**

Ensure you have collected:

- [ ] **Standard Training Results**:
  - [ ] BC on Needle Reach (success rate, training curves)
  - [ ] BC on Peg Transfer (success rate, training curves)
  - [ ] PPO on Needle Reach (success rate, training curves)
  - [ ] PPO failure documentation on Peg Transfer

- [ ] **Hyperparameter Sensitivity**:
  - [ ] BC network size sensitivity (64, 128, 256, 512)
  - [ ] BC learning rate sensitivity (1e-5 to 1e-3)
  - [ ] PPO+BC β sensitivity *(when implemented)*

- [ ] **Ablation Studies**:
  - [ ] PPO+BC with/without BC initialization *(when implemented)*
  - [ ] Different regularization strengths for BC

- [ ] **Sample Efficiency**:
  - [ ] BC with varying demo counts (25, 50, 100, 150)
  - [ ] Learning curves showing convergence rates

- [ ] **Statistical Validation**:
  - [ ] Multi-seed runs (minimum 5 seeds) for key results
  - [ ] Confidence intervals and significance tests

- [ ] **Analysis Artifacts**:
  - [ ] Training curves plots
  - [ ] Comparison tables (LaTeX format)
  - [ ] Statistical significance results
  - [ ] Failure analysis summaries

### 🎯 Expected Timeline

- **Phase 1-2** (Data + Standard): 1-2 days
- **Phase 3** (Hyperparameter Sensitivity): 2-3 days
- **Phase 4** (Create Optimized Configs): 0.5-1 day  
- **Phase 5-6** (Ablation + Sample Efficiency): 2-3 days
- **Phase 7** (Multi-seed): 2-3 days
- **Phase 8-9** (Analysis + Verification): 1-2 days

**Total Estimated Time**: 9-15 days of compute time

### 🔄 **Key Methodological Insight**:

**Hyperparameter Optimization First** → **Then Ablation/Sample Efficiency**

This ensures that ablation studies and sample efficiency analyses are conducted with optimal baseline parameters, leading to more reliable and interpretable results for your thesis.

This systematic approach will generate comprehensive results for your thesis Chapter 4, demonstrating thorough experimental validation and analysis.

## 🔬 Hyperparameter Optimization Methodology

### Why Use Automated Optimization?

The new `hyperopt_unified.py` script provides **significant advantages** over manual hyperparameter tuning:

1. **Efficiency**: Tree-structured Parzen Estimator (TPE) is ~10x more efficient than grid search
2. **Intelligence**: Learns from previous trials to suggest better hyperparameters
3. **Early Stopping**: Automatically prunes poor-performing trials to save compute time
4. **Reproducibility**: Complete experiment tracking with seeds and configurations
5. **Visualization**: Automatic generation of optimization plots for thesis figures

### Integration with 8-Phase Plan

The hyperparameter optimization seamlessly integrates with your systematic experimental plan:

**Replace Phase 3 (Manual Hyperparameter Sensitivity) with:**
```bash
# Phase 3: Automated Hyperparameter Optimization
python3 hyperopt_unified.py --stage sequential --task needle_reach --bc-trials 50 --ppo-bc-trials 30
python3 hyperopt_unified.py --stage sequential --task peg_transfer --bc-trials 50 --ppo-bc-trials 30
```

**Timeline Impact:**
- **Manual approach**: Phase 3 = 2-3 days compute time
- **Automated approach**: Phase 3 = 1-2 days compute time (with better results)
- **Total time saved**: 1-2 days while finding superior hyperparameters

### Sequential Optimization Workflow

1. **Stage 1 - BC Optimization**: Find optimal BC hyperparameters
   - Search space: Network architecture, learning rate, batch size, regularization
   - Optimize for: Maximum success rate on evaluation episodes
   - Output: Best BC model + optimal configuration

2. **Stage 2 - PPO+BC Integration**: Optimize hybrid algorithm using best BC
   - Search space: PPO parameters + BC loss weight + integration frequency
   - Fixed: BC model from Stage 1 (optimal foundation)
   - Optimize for: Maximum success rate improvement over BC alone

### Thesis Benefits

Using automated hyperparameter optimization provides several advantages for your thesis:

1. **Methodological Rigor**: Demonstrates state-of-the-art optimization techniques
2. **Statistical Validity**: Multiple trials with different hyperparameters strengthen conclusions
3. **Reproducibility**: Complete hyperparameter search history and reasoning
4. **Time Efficiency**: More experiments in less time = more comprehensive results
5. **Publication Quality**: Optuna visualizations make excellent thesis figures

### Quick Start for Phase 3 Replacement

Instead of manual sensitivity analysis, run:

```bash
# Install dependencies (if not already installed)
cd /home/lele/Desktop/Master-Project/summer_project
pip install -e .

# Generate expert demonstrations (Phase 1)
cd experiments
python3 generate_expert_data_needle_reach.py
python3 generate_expert_data_peg_transfer.py

# Run automated hyperparameter optimization (Phase 3 replacement)
python3 hyperopt_unified.py --stage sequential --task needle_reach \
  --bc-trials 50 --ppo-bc-trials 30 \
  --output-dir results/hyperopt

python3 hyperopt_unified.py --stage sequential --task peg_transfer \
  --bc-trials 50 --ppo-bc-trials 30 \
  --output-dir results/hyperopt

# Results will be in:
# - results/hyperopt/bc_needle_reach_*/optimal_config_bc_needle_reach.yaml
# - results/hyperopt/ppo_bc_needle_reach_*/optimal_config_ppo_bc_needle_reach.yaml
# - Plus comprehensive visualizations and analysis
```

This automated approach replaces the manual Phase 3 while providing superior hyperparameter discovery and comprehensive experimental documentation for your thesis.