# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a custom dVRK (da Vinci Research Kit) simulation environment built from scratch using PyBullet and Gymnasium. The project implements a standardized workflow for surgical robot learning combining Imitation Learning (IL) and Reinforcement Learning (RL). All development is containerized using Docker for reproducibility.

## Core Architecture

### Main Components
- **`src/dvrk_gym/`**: Core gym environment library
  - `envs/`: Environment implementations (NeedleReach, PegTransfer)
  - `robots/`: Robot models (PSM arm implementation)
  - `utils/`: Utilities for PyBullet, robotics, and wrappers
  - `assets/`: URDF models and meshes for robots and objects
- **`scripts/`**: Training and evaluation scripts
- **`archive/SurRoL/`**: Legacy reference implementation (not actively used)

### Environment Registration
Environments are registered in `src/dvrk_gym/envs/__init__.py`:
- `NeedleReach-v0`: Basic needle reaching task  
- `PegTransfer-v0`: Peg transfer task (modeled after SurRoL implementation)
- Additional environments can be registered following the same pattern

## Development Commands

### Docker-based Development (REQUIRED)
**IMPORTANT**: ALL development work must be done through Docker containers. Never run Python scripts directly on the host machine. This ensures reproducibility and proper environment setup.

```bash
# Start development environment
docker compose -f docker/docker-compose.yml up -d

# Run commands in container
docker compose -f docker/docker-compose.yml exec dvrk-dev <command>

# Stop development environment
docker compose -f docker/docker-compose.yml down
```

### Testing and Debugging
**CRITICAL**: When testing environments or running any Python code, always use the Docker container:
```bash
# Test environment creation
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 -c "import gymnasium as gym; import dvrk_gym; env = gym.make('PegTransfer-v0'); print('Environment test passed')"

# Interactive debugging shell
docker compose -f docker/docker-compose.yml exec dvrk-dev /bin/bash
```

### Standard Workflow Commands

1. **Generate Expert Data**:
   ```bash
   # For NeedleReach task
   docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/generate_expert_data.py
   
   # For PegTransfer task
   docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/generate_expert_data_peg_transfer.py
   ```

2. **Train Behavioral Cloning**:
   ```bash
   docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py
   ```

3. **Train Pure RL (PPO)**:
   ```bash
   docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py
   ```

4. **Train DAPG (RL fine-tuning)**:
   ```bash
   docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py
   ```

### Evaluation Commands

**For PPO/DAPG models** (use `evaluate.py`):
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate.py --model-path <path_to_model.zip> --flatten --dense-reward
```

**For BC models** (use `evaluate_bc.py`):
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py --model-path <path_to_model.zip>
```

### Monitoring and Debugging
```bash
# TensorBoard for training visualization
docker compose -f docker/docker-compose.yml exec dvrk-dev tensorboard --logdir logs

# Test environment functionality
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/test_peg_transfer.py
```

### Package Installation
```bash
# Install in editable mode
pip3 install -e .
```

## Important Technical Details

### Reward Systems
- **PPO**: MUST use dense rewards (`use_dense_reward=True`)
- **DAPG**: Uses sparse rewards to avoid reward hacking
- **BC**: Reward-agnostic (uses expert demonstrations)

### Observation Handling
- RL/DAPG training uses `FlattenDictObsWrapper` for Stable-Baselines3 compatibility
- Always use `--flatten` flag when evaluating RL models

### Key Dependencies
- `stable-baselines3~=2.2.1`: RL algorithms
- `imitation~=1.0.0`: Behavioral cloning and DAPG
- `pybullet`: Physics simulation
- `gymnasium`: Environment interface
- `protobuf==3.20.3`: Pinned for TensorBoard compatibility

### File Structure Patterns
- Environment classes inherit from `dvrk_gym.envs.dvrk_env.DVRKEnv`
- Robot implementations in `dvrk_gym.robots.psm.Psm`
- Assets follow standard PyBullet URDF structure in `assets/`
- Training logs saved to `logs/` with timestamped directories

### Expert Data Generation Pattern
- Each environment has its own dedicated expert data generation script
- Script naming: `generate_expert_data_<task_name>.py`
- Data files saved as: `data/expert_data_<task_name>.pkl`
- PegTransfer uses only successful episodes in the dataset

### Common Debugging Points
- Environment reward calculation in task-specific `_get_reward()` methods
- Robot kinematics and constraints in `robots/psm.py`
- Contact detection and grasping logic in individual task files
- URDF loading and asset paths in `assets/` directory

## PegTransfer Debugging Guide

### Critical Issues and Solutions When Mimicking SurROL

#### 1. TIP-EEF Offset Problem
**Issue**: Robot reaches waypoints but never activates grasping (activated=-1)

**Root Cause**: TIP is 5.1cm below EEF, but waypoints calculated for EEF position

**Detection**: 
- TIP-object distance stays >1cm (activation threshold)
- Robot appears to "grasp air" above objects

**Solution**: Subtract TIP-EEF offset from grasp waypoint heights:
```python
# In _define_waypoints()
grasp_height = pos_obj[2] + (0.003 + 0.0102) * self.SCALING - 0.051 - 0.013
```

#### 2. Activation vs Constraint Creation
**Issue**: Robot activates (activated=0) but doesn't physically grasp objects

**Root Cause**: `_meet_contact_constraint_requirement()` requires object height > goal+5cm

**Detection**:
- activated=0 but constraint=False
- Object height never changes during "grasping"

**Solution**: Create constraint immediately upon activation:
```python
def _meet_contact_constraint_requirement(self) -> bool:
    return self._activated >= 0  # Create constraint as soon as activated
```

#### 3. Contact Detection Settings
**Critical Settings for PegTransfer**:
```python
self._contact_approx = True   # Use distance-based activation (threshold: 1cm)
self._waypoint_goal = True    # PegTransfer has waypoint goals
```

#### 4. Precision Analysis Workflow
1. **Use debug scripts to measure exact distances**:
   - `debug_precision.py` - Analyze waypoint reaching precision
   - `debug_near_miss.py` - Find closest approach distance and offset vectors
   - `debug_activation.py` - Monitor activation thresholds
   - `debug_constraint.py` - Track constraint creation during grasping

2. **Key Measurements**:
   - TIP-to-object distance during grasping attempts
   - EEF position vs waypoint position accuracy
   - Activation threshold: `2e-3 * SCALING` (1cm for SCALING=5)

#### 5. Coordinate System Verification
- `pose_rcm2world()` returns EEF position, not TIP
- Observations use EEF coordinates (`obs['observation'][:3]`)
- Activation detection uses TIP coordinates
- This mismatch requires height compensation in waypoint calculations

#### 6. Debug Script Usage Pattern
```bash
# Step-by-step debugging approach:
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/debug_precision.py
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/debug_near_miss.py  
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/debug_activation.py
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/debug_constraint.py
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/debug_full_workflow.py
```

#### 7. Success Indicators
- **Activation Success**: activated=0, TIP-object distance <1cm
- **Constraint Success**: constraint=True, object height increases
- **Task Success**: terminated=True, object-goal distance <3cm  
- **Precision Target**: TIP-object distance <0.5mm during grasping

#### 8. Common Failure Patterns
- **"Grasping air"**: TIP-EEF offset not compensated in waypoints
- **"Touches but doesn't grab"**: Constraint creation conditions too strict
- **"Robot stuck in air"**: Workspace limits preventing downward movement
- **"Never activates"**: contact_approx=False requires physical contact vs distance-based

## Testing and Validation
- Use scripts in `scripts/debug_*.py` for specific debugging
- `scripts/comprehensive_test.py` for full environment validation
- Always test in Docker environment to ensure reproducibility