# Technical Context

## 1. Core Technology Stack
This project is built upon a modern, standardized stack for robot learning research.

| Category | Technology | Version/Spec | Rationale |
| :--- | :--- | :--- | :--- |
| **Simulator** | PyBullet | latest | Lightweight, fast, and directly supports the required URDF format. <br><br> **Configuration Notes:** <br> - `__init__` must define spaces. <br> - Camera must be reset after `p.resetSimulation`. <br> - Beware of position vs. `globalScaling` interactions. <br> - See `systemPatterns.md` for full details. |
| **Environment API** | Gymnasium | latest | The official standard for RL environments, ensuring broad compatibility. |
| **RL Algorithms** | Stable-Baselines3 | ~2.2.1 | A robust, well-maintained library for PyTorch-based RL algorithms. |
| **Imitation Learning** | imitation | ~1.0.0 | Designed to work seamlessly with Stable-Baselines3 for IL tasks. |
| **Deep Learning** | PyTorch | ~1.13.1 | The underlying framework for SB3 and `imitation`. |
| **Core Numerics** | NumPy | <2.0 | Pinned to avoid compatibility issues with older compiled libraries like `roboticstoolbox`. |
| **Containerization** | Docker | - | For creating a reproducible development environment. |
| **Base Image** | `nvidia/cudagl` | 11.4.2-base-ubuntu20.04 | Provides essential NVIDIA drivers and OpenGL for GUI applications inside Docker. |
| **Packaging** | `setuptools`, `pip` | latest | Using a `pyproject.toml` file with a `src` layout for a modern, installable package structure. |
| **GUI Forwarding** | X11 Forwarding | - | The container uses `network_mode: "host"` and shares the X11 socket to display GUI applications directly on the host. |

## 2. Deprecated Technologies
The following technologies were part of the initial project setup but have been **explicitly abandoned** due to compatibility and maintenance issues.

-   **SurRoL Framework**: The core dependency that was removed. Its architecture, based on TensorFlow 1.x and an outdated `gym` API, proved impossible to integrate with modern libraries due to fundamental graphics-level incompatibilities.
-   **TensorFlow (1.x)**: Replaced by PyTorch. The legacy TF 1.x stack was found to have an unresolvable conflict with modern NVIDIA drivers, leading to `Segmentation Faults` and `EGL_BAD_CONFIG` errors, making it a technical dead end.
-   **OpenAI `baselines`**: Replaced by Stable-Baselines3.
