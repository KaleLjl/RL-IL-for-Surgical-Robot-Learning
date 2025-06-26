# Active Context

## 1. Current Focus: Transitioning to DAPG
Our primary focus has shifted from debugging the pure RL agent to preparing for the next stage of the project: training a Demonstration-Augmented Policy Gradient (DAPG) agent. This follows the successful resolution of the PPO training bug.

### Summary of PPO Bug Resolution
- **Initial Problem**: The PPO agent, regardless of policy (`MultiInputPolicy` or `MlpPolicy`), failed to learn a meaningful strategy, resulting in a 0% success rate during evaluation.
- **Diagnostic Journey**:
    1.  Our initial hypothesis was a bug in Stable-Baselines3's `MultiInputPolicy` when handling `Dict` observation spaces. This led us to implement a **"Flattening Fallback"** using a `FlattenDictObsWrapper` and `MlpPolicy`.
    2.  When the flattened approach also failed, we discovered the true root cause: the **sparse reward function** was insufficient for a pure RL agent to learn effectively from scratch.
    3.  The final, successful solution involved implementing a **switchable reward system** in the environment. We trained the PPO agent using a **dense reward** (`reward = -distance`) combined with the **flattened observation space**. This produced a successful model with an 80% success rate.
- **Key Learnings**:
    -   The primary blocker for pure RL was the sparse reward signal.
    -   The necessity of the `FlattenDictObsWrapper` is now **unconfirmed**. While it was kept as a safety measure, it's possible that `MultiInputPolicy` could work correctly with a dense reward. This can be revisited later if needed.

## 2. Immediate Actions: Preparing for DAPG
With a functional pure RL training pipeline, we now proceed to the next logical step in our development workflow.

1.  **Documentation**:
    -   **Goal**: Solidify our learnings and provide clear instructions for future use.
    -   **Action**: Create a root-level `README.md` to serve as a user-facing manual, detailing standard training and evaluation commands. This includes specifying which evaluation script (`evaluate.py` vs. `evaluate_bc.py`) to use for which model type.
    -   **Action**: Update the Memory Bank (`activeContext.md`, `progress.md`, `systemPatterns.md`) to reflect the complete diagnostic journey and establish new patterns for reward design and evaluation.

2.  **Next Step: DAPG Training**:
    -   The next major technical task is to create and run the training script for DAPG (`scripts/train_dapg.py`).
    -   This will leverage our pre-trained BC policy and the sparse reward function in the environment, following the established workflow.
