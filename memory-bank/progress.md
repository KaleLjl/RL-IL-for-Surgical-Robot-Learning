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

## 3. Current Status (As of 2025-06-25)

-   **Phase**: **Environment Configuration Complete. Validation Starting.**
-   **What Works**:
    -   A modern, reproducible Docker development environment has been successfully built and verified.
    -   The project follows a clean `src`-based layout with a `pyproject.toml` for packaging.
    -   All core Python dependencies (PyTorch, SB3, imitation, PyBullet, roboticstoolbox-python) are installed and working.
    -   GUI forwarding via X11 has been successfully configured and tested.
    -   The `dvrk_gym` package has been successfully implemented, mirroring the structure of the original SurRoL robot and environment logic.
    -   The first environment, `dvrk_gym/NeedleReach-v0`, is now **fully configured and verified**. Through an iterative debugging process, it has been confirmed to be a 1:1 match of the original SurRoL environment in terms of initialization logic, physics, object/robot positioning, and visual appearance.
    -   Key learnings from the complex debugging process have been documented in `systemPatterns.md` to accelerate future environment development.
-   **What's Left**: The environment now needs to be validated by training a baseline learning agent to ensure it presents a solvable problem.
-   **Immediate Next Steps**:
    1.  Create a training script (`scripts/train_reach.py`) using Stable-Baselines3.
    2.  Train a simple RL agent (e.g., PPO or SAC) on the `dvrk_gym/NeedleReach-v0` environment.
    3.  Evaluate the trained agent's performance to validate that the environment is stable and learnable.
