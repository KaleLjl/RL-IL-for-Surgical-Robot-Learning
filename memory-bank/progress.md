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

## 4. ML Validation Phase 2: Reinforcement Learning
- **Status**: **Complete.**
- **Outcome**: Successfully trained a PPO agent with an 80% success rate after a complex debugging process. This validated the pure RL portion of our development workflow.
- **Evolution of the "Constant Action" Bug**:
    - **Initial State**: The PPO agent failed to learn, producing a 0% success rate.
    - **Investigation**: We initially suspected a bug in SB3's `MultiInputPolicy` and implemented a `FlattenDictObsWrapper` as a workaround. When this also failed, we analyzed the difference between training (`deterministic=False`) and evaluation (`deterministic=True`) behavior.
    - **Root Cause**: The final root cause was identified as an improper **sparse reward** function for a pure RL agent. The lack of a guiding signal caused the agent to converge on a suboptimal "do nothing" policy.
    - **Resolution**: A **dense reward** system (`reward = -distance`) was implemented, selectable via an environment flag. Training with the combination of dense rewards and the flattened observation space proved successful.
    - **Open Question**: The necessity of the `FlattenDictObsWrapper` is now unconfirmed, as the primary issue was the reward signal. This is noted for potential future simplification.

## 5. ML Validation Phase 3: Demonstration-Augmented Policy Gradient (DAPG)
- **Status**: **Pending.**
- **Next Steps**:
    1.  Create and/or verify the `train_dapg.py` script.
    2.  This script will use the successful BC model as a pre-trained starting point.
    3.  Crucially, it will use the environment's **sparse reward** function, which is appropriate for this algorithm, as per our established `reward-system-guidelines`.
    4.  Document the entire workflow in a user-facing `README.md`.
