# dVRK Gym: Surgical Robot Control via Machine Learning

A Gymnasium environment for training machine learning algorithms on da Vinci Research Kit (dVRK) surgical manipulation tasks using PyBullet simulation.

**Master's Thesis Project**: "Autonomous Surgical Robot Control Through Machine Learning"  
**Author**: Jiale Li  
**Email**: lijiale77kl@gmail.com  
**Program**: MSc AI and Robotics, University College London (UCL)  
**Year**: 2024-2025  

## Overview

This project implements and compares three machine learning approaches for controlling surgical robots in simulation:

- **Behavioral Cloning (BC)**: Supervised learning from expert demonstrations
- **Proximal Policy Optimization (PPO)**: Reinforcement learning approach  
- **PPO+BC**: Hybrid approach combining RL with imitation learning guidance

The environments simulate two fundamental surgical tasks using the da Vinci Patient Side Manipulator (PSM) robot in PyBullet physics simulation.

### 🔄 Relation to SurRoL

This project is a **modernized implementation** inspired by [SurRoL](https://github.com/med-air/SurRoL) (Surgical Robot Learning) with significant technological improvements:

**Key Upgrades from SurRoL:**
- **🆕 Modern ML Stack**: Migrated from OpenAI Gym to Gymnasium (actively maintained)
- **🆕 Latest Libraries**: Updated to Stable-Baselines3 v2.2+ and Imitation v1.0+
- **🆕 Hybrid Algorithms**: Added PPO+BC implementation for improved performance
- **🆕 Container Support**: Full Docker integration with GPU acceleration
- **🆕 Advanced Optimization**: Optuna-based hyperparameter optimization
- **🆕 Robust Evaluation**: Comprehensive metrics including noise robustness testing

**Why This Version:**
- SurRoL uses deprecated dependencies (OpenAI Gym, older SB3 versions)
- This implementation focuses on systematic algorithm comparison and reproducible research
- Enhanced experimental framework with modern best practices for ML research

## Surgical Tasks

### 1. Needle Reach (`NeedleReach-v0`)
- **Objective**: Precisely position the surgical instrument tip at a target location
- **Control**: 6-DOF end-effector positioning
- **Success Criteria**: Reach within 5mm of target position
- **Episode Length**: 100 steps

### 2. Peg Transfer (`PegTransfer-v0`)  
- **Objective**: Grasp colored blocks and transfer them between pegs
- **Control**: 5-DOF positioning + gripper control
- **Success Criteria**: Successfully place block on destination peg
- **Episode Length**: 150 steps


## Installation

### 🐳 Docker Setup

This project requires Docker with GPU support for optimal performance:

```bash
# Prerequisites: Docker, docker-compose, and NVIDIA Container Runtime
# Install NVIDIA Container Runtime: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Clone this project 
git clone https://github.com/KaleLjl/RL-IL-for-Surgical-Robot-Learning

# Build and start the container
cd docker
docker-compose up -d

# Enter the container
docker exec -it dvrk_dev bash

# Verify installation inside container
python3 -c "import dvrk_gym; print('dVRK Gym installed successfully')"
```

**Benefits**: Pre-configured CUDA 11.4, OpenGL, dependencies, and GPU acceleration in a consistent containerized environment.


## Quick Start (Using Docker 🐳)

**All commands below should be run inside the Docker container**:

```bash
# Start the container (if not already running)
cd docker && docker-compose up -d

# Enter the container
docker exec -it dvrk_dev bash

# Now you're inside the container at /app directory
```

### 1. Generate Expert Demonstrations
```bash
cd experiments

# Generate demonstrations for needle reach task (50 episodes by default)
python3 generate_expert_data_needle_reach.py --num_episodes 100

# Generate demonstrations for peg transfer task (50 episodes by default)
python3 generate_expert_data_peg_transfer.py --num_episodes 100
```

### 2. Train Models

**Behavioral Cloning**:
```bash
# Train BC on needle reach (saves model to results directory)
python3 train_bc.py --env NeedleReach-v0 --output-dir results/bc_needle_reach

# Train BC on peg transfer
python3 train_bc.py --env PegTransfer-v0 --output-dir results/bc_peg_transfer
```

**PPO (Reinforcement Learning)**:
```bash
# Train PPO on needle reach (successful, ~3 hours, 100k timesteps)
python3 train_ppo.py --env NeedleReach-v0 --output-dir results/ppo_needle_reach

# Train PPO on peg transfer (warning: typically fails, 300k timesteps)
python3 train_ppo.py --env PegTransfer-v0 --output-dir results/ppo_peg_transfer
```

**PPO+BC (Hybrid Approach)**:
```bash
# Note: PPO+BC implementation available but requires manual configuration
# Check the script for detailed usage instructions
python3 train_ppo+bc.py --help
```

### 3. Evaluate Trained Models
```bash
# Evaluate BC model (environment auto-detected from model path)
python3 evaluate_bc.py --model results/bc_needle_reach/bc_needle_reach_*.zip --episodes 100 --output-dir results/evaluation

# Test robustness with action noise (0.1 std deviation)
python3 evaluate_bc.py --model results/bc_needle_reach/bc_needle_reach_*.zip --episodes 100 --action-noise-test --noise-std 0.1

# Save evaluation videos
python3 evaluate_bc.py --model results/bc_needle_reach/bc_needle_reach_*.zip --episodes 5 --save-video --video-dir videos
```

### 4. Hyperparameter Optimization
```bash
# Automated hyperparameter optimization using Optuna (check script for available options)
python3 experiment_hyperopt_unified.py --help
```


## Project Structure

```
summer_project/
├── README.md                    # This file
├── pyproject.toml              # Package configuration
├── src/dvrk_gym/               # Main package
│   ├── envs/                   # Gymnasium environments
│   │   ├── needle_reach.py     # Needle reach task
│   │   └── peg_transfer.py     # Peg transfer task
│   ├── robots/                 # Robot models and control
│   │   └── psm.py             # Patient Side Manipulator
│   ├── assets/                 # URDF models and meshes
│   └── utils/                  # Utility functions
├── experiments/                # Training and evaluation scripts
│   ├── train_bc.py            # Behavioral Cloning training
│   ├── train_ppo.py           # PPO training  
│   ├── train_ppo+bc.py        # Hybrid PPO+BC training
│   ├── evaluate_*.py          # Evaluation scripts
│   ├── generate_expert_*.py   # Expert data generation
│   ├── data/                  # Expert demonstration data
│   └── results/               # Trained models and logs
└── docker/                    # Containerized environment
```


## Algorithm Overview

- **BC**: Supervised learning from demonstrations (2-layer MLP, Adam optimizer)
- **PPO**: On-policy RL with actor-critic architecture (fails on complex manipulation)  
- **PPO+BC**: Hybrid approach combining PPO with BC guidance (α=0.05 weighting)

## Advanced Docker Usage

```bash
# Container management
docker-compose up -d                    # Start container
docker-compose down                     # Stop container
docker-compose logs -f dvrk-dev        # View logs

# Background training
docker exec -d dvrk_dev python3 experiments/train_ppo.py --env NeedleReach-v0 --output-dir results/ppo_needle_reach
docker exec dvrk_dev tail -f results/ppo_needle_reach/logs/training.log  # Monitor progress
```


## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{li2025surgical,
  title={Autonomous Surgical Robot Control Through Machine Learning},
  author={Li, Jiale},
  school={University College London},
  year={2025},
  type={{MSc} Thesis},
  department={Department of Computer Science}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UCL Robotics Group** for supervision and computational resources
- **SurRoL Team** ([MedAIR Lab](https://github.com/med-air/SurRoL)) for the original surgical robot learning framework that inspired this work
- **da Vinci Research Community** for surgical robotics insights  
- **Open Source Libraries**: Stable-Baselines3, Imitation, PyBullet, and Gymnasium teams

---

**Contact**: For questions about this research, please contact Jiale Li at lijiale77kl@gmail.com