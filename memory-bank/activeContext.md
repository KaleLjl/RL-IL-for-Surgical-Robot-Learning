# Active Context

## 1. Current Focus: Reinforcement Learning Implementation
With the successful training and evaluation of the Behavioral Cloning (BC) agent (94% success rate), the project's focus now shifts to the next major milestone: **implementing a Reinforcement Learning (RL) pipeline.**

### The "imitation" Library Bug & The Solution
After a lengthy and systematic debugging process, we conclusively proved that the `imitation` library (v1.0.1) has a fundamental bug in its handling of `gymnasium.spaces.Dict` observation spaces. The library's internal validation logic incorrectly calculates the length of dictionary-based observations, leading to persistent `ValueError` exceptions.

The final, robust solution was to **circumvent the bug by flattening the `Dict` observations into a single `Box` (NumPy array) space at the data-loading stage**. This was implemented in the `train_bc.py` script. By feeding the `imitation.BC` trainer a simple, flat array bottlene, we avoided all of the library's problematic internal data processing logic and successfully trained the agent.

This entire debugging journey has been a critical learning experience and has been documented in `systemPatterns.md`.

## 2. Next Immediate Actions
The successful BC evaluation provides a strong baseline. The project will now proceed with the following core ML objectives:

1.  **Implement RL Training Pipeline**: This is the top priority. The `scripts/train_rl.py` file needs to be developed to train an agent from scratch using a suitable algorithm from Stable-Baselines3 (e.g., PPO or SAC) on the `NeedleReach-v0` environment.
2.  **Compare RL vs. BC Performance**: Once a satisfactory RL agent is trained, its performance (success rate, sample efficiency, final reward) will be benchmarked against the 94% success rate achieved by the BC agent.
3.  **Hyperparameter Tuning**: Depending on the initial RL results, we may need to perform hyperparameter tuning for either or both the RL and BC models to maximize performance.

## 3. Key Learnings & Decisions
-   **Environment Replication is Deceptive**: Replicating a PyBullet environment requires a meticulous, multi-layered approach. The process revealed several non-obvious pitfalls that are now documented in `systemPatterns.md` under "PyBullet Environment Configuration Patterns". This documentation is critical for efficiently configuring future tasks.
-   **Control is paramount**: Owning the entire environment stack, from simulation to the learning algorithm, gives us the control needed to debug effectively and build a stable platform. This decision was validated by the successful debugging of the `NeedleReach-v0` environment.
-   **SurRoL is a technical dead end**: The VNC experiment provided the final, conclusive evidence. The framework's reliance on a legacy graphics stack (TF 1.x, CUDA 10.0) is fundamentally incompatible with modern NVIDIA drivers, leading to unresolvable `EGL_BAD_CONFIG` errors. Further attempts to patch it are a poor use of time.
-   **PyBullet over Isaac Sim**: While Isaac Sim is powerful, its complexity and high barrier to entry make it unsuitable for our immediate goal of rapid prototyping and algorithm validation. PyBullet offers the best balance of performance, ease of use, and flexibility for this project.
-   **Flattening Data is a Robust Fallback**: When a library has buggy or poorly documented support for complex data structures like `Dict` spaces, the most reliable solution is to preprocess the data into a simple, flat format (`Box` space) before passing it to the library. This circumvents internal bugs and is a valuable, pragmatic engineering pattern.
