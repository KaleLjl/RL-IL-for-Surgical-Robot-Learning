# Project Progress & Evolution

## 1. Initial Phase: The SurRoL Migration (Abandoned)

-   **Objective**: The project began with the goal of migrating the TensorFlow 1.x-based SurRoL framework to a modern PyTorch and Stable-Baselines3 (SB3) stack.
-   **Actions Taken**:
    -   A new Docker environment was created.
    -   Numerous dependency issues were resolved (`build-essential`, `EGL_BAD_CONFIG`).
    -   The `train_bc.py` script was rewritten for SB3.
-   **Outcome**: The migration effort was ultimately **abandoned**. The root cause was a fundamental incompatibility between the legacy stack (`TensorFlow 1.14`, `CUDA 10.0`) and modern host NVIDIA drivers.
    -   **Initial Blocker**: Persistent `Segmentation fault` errors originating from the host's graphics driver (`libnvidia-glcore.so`).
    -   **Final Diagnosis**: A VNC-based experiment resolved the segmentation fault but revealed a deeper, unresolvable `EGL_BAD_CONFIG` error (12292). This conclusively proved that the old libraries could not establish a hardware-accelerated rendering context with the modern host, making the entire stack a technical dead end.

## 2. Strategic Pivot: Custom Environment Development

-   **Decision**: Based on the conclusive findings from the VNC experiment, a key strategic decision was made to **stop fixing SurRoL and instead build a new, custom dVRK environment from scratch**.
-   **Rationale**: This approach, centered on PyTorch and Stable-Baselines3, provides full control over the technology stack, eliminates the root cause of the driver and EGL incompatibilities, and ensures long-term stability. It shifts the project's focus from reverse-engineering and patching to constructive, forward-looking development.

## 3. ML Validation Phase: Overcoming the `imitation` Bug

-   **Initial Blocker**: The project was significantly delayed by a fundamental bug in the `imitation` library's handling of `Dict` observation spaces. The library's internal validation logic consistently failed, misinterpreting the structure of our correctly formatted expert data.
-   **Debugging & Diagnosis**:
    -   A systematic process of elimination was followed, including creating a minimal, reproducible example (`diagnose_imitation_bug.py`).
    -   This process definitively proved the issue was not in our code but within the `imitation` library itself. We confirmed we were using the latest version (`1.0.1`), which had not fixed the bug.
-   **Resolution**: After exploring multiple complex workarounds, the most robust and pragmatic solution was chosen: **flatten the `Dict` observation data into a simple `Box` (array) format at the data-loading stage**.
    -   The `train_bc.py` script was refactored to perform this flattening before passing the data to the `imitation` trainer.
    -   This approach successfully circumvented the library's buggy validation logic.
-   **Outcome**: **The ML validation blocker is now resolved.** We have successfully trained a Behavioral Cloning agent on the `NeedleReach-v0` task and saved the resulting model.

## 4. Current Status (As of 2025-06-25)

-   **Phase**: **ML Validation & Iteration.**
-   **What Works**:
    -   A modern, reproducible Docker development environment.
    -   A fully configured and verified `dvrk_gym/NeedleReach-v0` environment, compliant with Gymnasium API standards.
    -   A robust, working pipeline for training Behavioral Cloning (BC) agents, achieving a 94% success rate.
    -   A clear pattern for handling library bugs with complex data types: preprocess them into the simplest possible format.
    -   **A custom, in-house implementation of a DAPG-style algorithm (`PPOWithBCLoss`)**, which inherits from `stable-baselines3.PPO` and adds a behavioral cloning loss term.
    -   A working training script (`scripts/train_dapg.py`) that successfully launches and monitors the custom DAPG training process using TensorBoard.
-   **What's Left**:
    -   Full evaluation of the trained DAPG model.
    -   Systematic comparison between the BC and DAPG policies.
    -   Hyperparameter tuning for the DAPG algorithm (e.g., `bc_loss_weight`).
    -   Implementation of the next stage of our standardized workflow: Residual Reinforcement Learning (RRL).
-   **Immediate Next Steps**:
    1.  **Monitor and Complete DAPG Training**: Allow the current DAPG training run to complete, monitoring its progress via TensorBoard.
    2.  **Evaluate DAPG Policy**: Create an evaluation script for the newly trained DAPG model to measure its success rate and compare it to the 94% BC baseline.
    3.  **Analyze and Tune**: Based on the evaluation results, analyze the policy's behavior and decide if hyperparameter tuning is necessary.
    4.  **Document Findings**: Update the memory bank with the results of the DAPG experiment.
