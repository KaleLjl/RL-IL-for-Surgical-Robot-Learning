# dVRK Gym Environment for Surgical Robot Learning

## 1. Project Overview

This project provides a stable, modern, and maintainable simulation environment for the da Vinci Research Kit (dVRK) surgical robot, built using PyBullet and Gymnasium. It is designed to facilitate research and development in robot learning, with a focus on a standardized workflow combining Imitation Learning (IL) and Reinforcement Learning (RL).

The entire development environment is containerized using Docker to ensure reproducibility.

## 2. Standard Development Workflow

The project follows a standardized, multi-stage workflow for developing policies.

1.  **Collect Expert Data**: Generate a dataset of expert demonstrations.
    - `docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/generate_expert_data.py`
2.  **Train BC Policy**: Train a baseline policy using Behavioral Cloning.
    - `docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_bc.py`
3.  **Train RL Policy**: Train a pure RL agent from scratch.
    - `docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_rl.py`
4.  **Train DAPG Policy**: Fine-tune the BC policy using RL.
    - `docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/train_dapg.py`

## 3. How to Monitor Training (TensorBoard)

To visualize training progress, including rewards, loss functions, and other metrics, run TensorBoard in a separate terminal.

**Command:**
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev tensorboard --logdir logs
```
This will start a web server (usually on `http://localhost:6006`) that you can open in your browser to view the live training curves.

## 4. How to Evaluate Models

Different models must be evaluated with their corresponding specialized scripts.

### Evaluating PPO or DAPG Models

Use the `evaluate.py` script for models trained with Reinforcement Learning (PPO, DAPG).

**Command Template:**
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate.py --model-path <path_to_your_model.zip> --flatten --dense-reward
```

**Example:**
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate.py --model-path logs/ppo_needle_reach_1750938298/run_1/checkpoints/rl_model_100000_steps.zip --flatten --dense-reward
```

**Key Flags:**
*   `--model-path`: **(Required)** Path to the `.zip` file of the trained model.
*   `--flatten`: **(Required for PPO/DAPG)** Must be included, as these models are trained on flattened observations.
*   `--dense-reward`: **(Recommended for PPO)** Use this to see meaningful reward values during evaluation, consistent with how the PPO model was trained.

### Evaluating Behavioral Cloning (BC) Models

Use the `evaluate_bc.py` script for models trained *only* with Behavioral Cloning.

**Command Template:**
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py --model-path <path_to_your_bc_model.zip>
```

**Example:**
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py --model-path models/bc_needle_reach_1750948490.zip
```

## 4. Important Notes for Developers

-   **Reward System**: The environment supports both **sparse** and **dense** rewards.
    -   **Pure RL (PPO)**: **Must** be trained with the **dense reward** function. The `train_rl.py` script handles this automatically by passing `use_dense_reward=True` to the environment.
    -   **DAPG**: Should be trained with the default **sparse reward** to avoid reward hacking.
-   **Observation Flattening**: For compatibility and stability with the Stable-Baselines3 library, all RL and DAPG training uses a `FlattenDictObsWrapper`. This is handled automatically in the training scripts.
