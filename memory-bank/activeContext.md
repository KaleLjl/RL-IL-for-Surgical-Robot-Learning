# Active Context

## 1. Current Focus: Reinforcement Learning Implementation
With the successful training and evaluation of the Behavioral Cloning (BC) agent (94% success rate), the project's focus now shifts to the next major milestone: **implementing a Reinforcement Learning (RL) pipeline.**

### The "imitation" Library Bug & The Solution
After a lengthy and systematic debugging process, we conclusively proved that the `imitation` library (v1.0.1) has a fundamental bug in its handling of `gymnasium.spaces.Dict` observation spaces. The library's internal validation logic incorrectly calculates the length of dictionary-based observations, leading to persistent `ValueError` exceptions.

The final, robust solution was to **circumvent the bug by flattening the `Dict` observations into a single `Box` (NumPy array) space at the data-loading stage**. This was implemented in the `train_bc.py` script. By feeding the `imitation.BC` trainer a simple, flat array bottlene, we avoided all of the library's problematic internal data processing logic and successfully trained the agent.

This entire debugging journey has been a critical learning experience and has been documented in `systemPatterns.md`.

## 2. Next Immediate Actions & Standardized Workflow
Based on the success of the BC agent, we have established a standardized, three-stage workflow for developing and refining policies. This workflow will be applied first to the `NeedleReach-v0` task and then serve as a blueprint for all future tasks.

1.  **Stage 1: Behavioral Cloning (BC) - *Completed for NeedleReach***
    -   **Goal**: Train a baseline policy via imitation learning.
    -   **Status**: A BC agent with a 94% success rate has been successfully trained and evaluated. This serves as our performance baseline.

2.  **Stage 2: Demonstration-Augmented Policy Gradient (DAPG) - *Training in Progress***
    -   **Goal**: Enhance the BC policy's robustness and performance by fine-tuning it with RL.
    -   **Implementation**: We have built a custom DAPG-style algorithm, `PPOWithBCLoss`, which integrates a behavioral cloning loss into the standard PPO training loop from `stable-baselines3`.
    -   **Status**: The training for the `NeedleReach-v0` task is currently running. The process is being monitored via TensorBoard, and the custom `bc_loss` is being successfully logged, validating our implementation.
    -   **Immediate Task**: Monitor the training until completion, then create an evaluation script to assess the performance of the resulting model against the BC baseline.

3.  **Stage 3: Residual Reinforcement Learning (RRL) - *Future Work***
    -   **Goal**: Further improve performance and safety by learning a small "residual" correction on top of a high-quality base policy.
    -   **Plan**: Once a satisfactory DAPG policy is trained, it will serve as the base controller for an RRL agent. This approach is expected to offer the highest potential for safe and effective Sim-to-Real transfer.

This structured approach (BC -> DAPG -> RRL) provides a clear, systematic, and extensible path for tackling complex robotic manipulation tasks.

## 3. Key Learnings & Decisions
-   **Environment Replication is Deceptive**: Replicating a PyBullet environment requires a meticulous, multi-layered approach. The process revealed several non-obvious pitfalls that are now documented in `systemPatterns.md` under "PyBullet Environment Configuration Patterns". This documentation is critical for efficiently configuring future tasks.
-   **Control is paramount**: Owning the entire environment stack, from simulation to the learning algorithm, gives us the control needed to debug effectively and build a stable platform. This decision was validated by the successful debugging of the `NeedleReach-v0` environment.
-   **SurRoL is a technical dead end**: The VNC experiment provided the final, conclusive evidence. The framework's reliance on a legacy graphics stack (TF 1.x, CUDA 10.0) is fundamentally incompatible with modern NVIDIA drivers, leading to unresolvable `EGL_BAD_CONFIG` errors. Further attempts to patch it are a poor use of time.
-   **PyBullet over Isaac Sim**: While Isaac Sim is powerful, its complexity and high barrier to entry make it unsuitable for our immediate goal of rapid prototyping and algorithm validation. PyBullet offers the best balance of performance, ease of use, and flexibility for this project.
-   **Flattening Data is a Robust Fallback**: When a library has buggy or poorly documented support for complex data structures like `Dict` spaces, the most reliable solution is to preprocess the data into a simple, flat format (`Box` space) before passing it to the library. This circumvents internal bugs and is a valuable, pragmatic engineering pattern.
