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

## 3. Current Status (As of 2025-06-24)

-   **Phase**: **Setup & Foundation**.
-   **What Works**: A clear technical direction and development plan has been established and agreed upon.
-   **What's Left**: The entire custom environment (`dvrk_gym`) needs to be built.
-   **Immediate Next Steps**:
    1.  Clean up the project repository by removing or archiving the old SurRoL-related files.
    2.  Implement the new `Dockerfile` to create a stable, VNC-enabled development environment based on the agreed-upon technology stack.
