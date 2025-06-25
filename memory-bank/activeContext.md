# Active Context

## 1. Current Focus
**Root Cause Identified.** The robotic arm's failure to move is caused by a faulty Inverse Kinematics (IK) implementation. The new `dvrk_gym` code replaced PyBullet's native IK solver with one from the `roboticstoolbox` library, which fails to find a valid solution and causes the robot to be commanded to its current position (i.e., zero movement).

## 2. Next Immediate Actions
The plan is to revert to the original, functional IK implementation from the SurRoL project.

1.  **Restore PyBullet IK**: Modify `src/dvrk_gym/robots/arm.py` to re-implement the `inverse_kinematics` method using PyBullet's native `p.calculateInverseKinematics`, matching the original SurRoL code.
2.  **Remove Faulty IK**: Delete the overriding `inverse_kinematics` method from `src/dvrk_gym/robots/psm.py` to allow it to inherit the correct implementation from the `Arm` base class.
3.  **Verify Fix**: Instruct the user to re-run the `scripts/test_direct_control.py` script to confirm that the robot now moves as expected.
4.  **Full Validation**: Once the direct control test passes, run the original `scripts/generate_expert_data.py` script to ensure the initial problem is fully resolved.
5.  **Document Final Outcome**: Update the memory bank with the results of the fix.

## 3. Key Learnings & Decisions
-   **Environment Replication is Deceptive**: Replicating a PyBullet environment requires a meticulous, multi-layered approach. The process revealed several non-obvious pitfalls that are now documented in `systemPatterns.md` under "PyBullet Environment Configuration Patterns". This documentation is critical for efficiently configuring future tasks.
-   **Control is paramount**: Owning the entire environment stack, from simulation to the learning algorithm, gives us the control needed to debug effectively and build a stable platform. This decision was validated by the successful debugging of the `NeedleReach-v0` environment.
-   **SurRoL is a technical dead end**: The VNC experiment provided the final, conclusive evidence. The framework's reliance on a legacy graphics stack (TF 1.x, CUDA 10.0) is fundamentally incompatible with modern NVIDIA drivers, leading to unresolvable `EGL_BAD_CONFIG` errors. Further attempts to patch it are a poor use of time.
-   **PyBullet over Isaac Sim**: While Isaac Sim is powerful, its complexity and high barrier to entry make it unsuitable for our immediate goal of rapid prototyping and algorithm validation. PyBullet offers the best balance of performance, ease of use, and flexibility for this project.
