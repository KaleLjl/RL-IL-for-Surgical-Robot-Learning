# Project Progress & Evolution

## 1. Initial Phase: The SurRoL Migration (Abandoned)
- **Status**: Complete.
- **Outcome**: Conclusively proved that the legacy SurRoL framework is incompatible with modern hardware and drivers. This was a necessary, valuable, and successful diagnostic effort that prevented wasted time and informed the project's strategic pivot.

## 2. Strategic Pivot: Custom Environment Development
- **Status**: Complete.
- **Outcome**: A fully functional, custom `dvrk_gym` environment was built from scratch, providing a stable, modern foundation for all subsequent ML work.

## 3. ML Validation Phase 1: Behavioral Cloning
- **Status**: Complete.
- **Outcome**: Successfully trained a Behavioral Cloning (BC) agent using the `imitation` library. This process involved diagnosing and circumventing a major bug in the library's handling of `Dict` observation spaces by implementing a data-flattening strategy. This success validated the custom environment and our debugging capabilities.

## 4. ML Validation Phase 2: Reinforcement Learning (Current)
- **Status**: **Blocked.**
- **What Works**:
    - A robust directory structure for managing experiments, runs, and checkpoints has been established and documented.
    - The PPO training script (`train_rl.py`) runs successfully, generating logs that *appear* to show learning (rising rewards, etc.).
    - A generic evaluation script (`evaluate.py`) has been created.
- **The Blocker: The "Constant Action" Bug**:
    - **Symptom**: The trained PPO model, despite positive training metrics, achieves 0% success in evaluation.
    - **Root Cause Analysis**: The agent is observed to perform the same action repetitively, regardless of the goal. This indicates the policy has degenerated and is ignoring observation inputs.
    - **Hypothesis**: There is a fundamental issue in how the `MultiInputPolicy` is handling the `Dict` observation space from our environment.
- **Immediate Next Steps**:
    1.  **Diagnose via MRE**: Create a minimal script to prove that the model produces the same action for different observations.
    2.  **Isolate the Fault**: Based on the MRE result, investigate the data pipeline from the environment's `_get_obs()` method through the policy's forward pass to find where the `desired_goal` information is being lost or ignored.
