# Project Brief: Custom dVRK-Gym Environment

## 1. Core Objective
The primary goal of this project is to develop a stable, modern, and maintainable simulation environment for the da Vinci Research Kit (dVRK) surgical robot. This environment will serve as a foundation for research and development in robot learning, with a focus on imitation learning (IL) and reinforcement learning (RL).

## 2. Motivation & Pivot
The project initially attempted to migrate the existing SurRoL framework to a modern tech stack (PyTorch, Stable-Baselines3). However, this effort was abandoned due to insurmountable dependency and compatibility issues stemming from SurRoL's reliance on legacy libraries (TensorFlow 1.x, older `gym`, and an incompatible PyBullet implementation).

To mitigate these risks and ensure long-term stability, a strategic pivot was made: **we will build a new environment from scratch.**

## 3. Key Goals
- **Develop a custom `dvrk_gym` library** that provides a standard Gymnasium interface for the dVRK robot.
- **Utilize PyBullet** for physics simulation, leveraging existing URDF models for the dVRK.
- **Ensure seamless integration** with modern RL/IL libraries, specifically Stable-Baselines3 and `imitation`.
- **Create a robust, containerized development environment** using Docker, complete with a virtual desktop for GUI-based debugging.
- **Implement and validate** benchmark surgical robotics tasks, starting with "Needle Reach".
