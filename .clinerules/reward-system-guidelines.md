## Brief overview
This document provides project-specific guidelines for designing and implementing reward systems in our custom `dvrk_gym` environments. The core principle is that the reward function must be tailored to the specific learning algorithm being used.

## Reward System Design Philosophy
- **One Size Does Not Fit All**: A single reward function is often insufficient for the entire policy development workflow (e.g., pure RL vs. imitation-based RL). Environments should be designed to support multiple reward schemes.
- **Clarity Over Complexity**: The purpose of each reward scheme should be clearly documented.

## Algorithm-Specific Reward Requirements

- **Pure Reinforcement Learning (e.g., PPO from scratch)**
  - **Requirement**: **Must** use a **dense reward** function.
  - **Rationale**: Pure RL agents learn via trial and error and have no prior knowledge. A dense reward (e.g., negative distance to the goal) provides a continuous learning signal, guiding the agent's exploration. Without it, the agent is learning "blind" and will likely fail to discover the correct behavior in a reasonable amount of time.
  - **Trigger Case**: When training a policy from a random initialization.

- **Imitation Learning (e.g., Behavioral Cloning - BC)**
  - **Requirement**: The reward system is **not used** during training.
  - **Rationale**: BC is a supervised learning method that learns by minimizing the difference between its actions and a provided set of expert actions. It does not use environmental rewards.

- **Imitation-Augmented RL (e.g., DAPG)**
  - **Requirement**: **Should** use a **sparse reward** function (e.g., `0` for success, `-1` for failure).
  - **Rationale**: These algorithms are initialized with a strong prior from expert demonstrations. The primary goal is to achieve the task objective, which a sparse reward defines unambiguously. Using a dense reward can sometimes lead to "reward hacking," where the agent deviates from the expert's style to exploit minor flaws in the dense reward metric.
  - **Trigger Case**: When fine-tuning a pre-trained BC policy.

## Implementation Pattern
- To support multiple reward schemes, the environment's `__init__` method should include a boolean flag to switch between them.
- **Example**:
  ```python
  def __init__(self, use_dense_reward: bool = False):
      self.use_dense_reward = use_dense_reward
      ...
  
  def _get_reward(self, obs):
      if self.use_dense_reward:
          return self._get_dense_reward(obs)
      else:
          return self._get_sparse_reward(obs)
  ```
- The training script is then responsible for enabling the correct mode (e.g., `gym.make("MyEnv-v0", use_dense_reward=True)` for pure RL).
