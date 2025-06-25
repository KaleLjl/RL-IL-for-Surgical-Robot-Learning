## Brief overview
This file documents key debugging philosophies and troubleshooting patterns derived from solving a complex library bug. These rules are designed to help future Cline avoid dead-end debugging cycles and make more effective strategic decisions when faced with fundamental blockers.

## Debugging Philosophy
- When encountering a paradoxical error (i.e., an error that contradicts verified facts), do not assume it's a simple logic mistake. Immediately suspect a deeper issue, such as a library bug, a version incompatibility, or a flawed assumption about how the library works.
- Prioritize creating a Minimal, Reproducible Example (MRE) to isolate the problem. This is the most effective way to prove or disprove a hypothesis about a library bug and is more efficient than making iterative changes to the main codebase.

## Data and Model Matching Principle
- A core principle of ML development is that the **data format must strictly match the policy/model that consumes it**.
- When you transform the data (e.g., flattening a `Dict` observation to a `Box`), you **must** also change the policy accordingly (e.g., from `MultiInputActorCriticPolicy` to `MlpPolicy`). A mismatch between data structure and policy architecture is a common source of errors.

## The "Flattening Fallback" Pattern
- When a third-party library shows buggy or poorly documented support for complex data structures (like `gym.spaces.Dict`), the most robust solution is often to **circumvent the bug by simplifying the data**.
- **Trigger Case**: If you have confirmed your data is correct but the library's data loading or validation logic repeatedly fails, stop trying to fix the data pipeline.
- **Action**: Preprocess the complex data into the simplest possible flat format (e.g., a single NumPy array / `Box` space) before passing it to the library. This is a powerful, pragmatic fallback strategy.

## Strategic Pivoting
- Do not spend excessive time fighting a library bug. If a core library proves to be a fundamental blocker, proactively propose a strategic pivot to the user.
- Clearly present the alternative options, such as:
    1.  **Circumventing**: Modifying the project's design to avoid the buggy feature (e.g., flattening the observation space).
    2.  **Replacing**: Finding an alternative library.
    3.  **Fixing**: Forking the library to fix the bug directly.
- Discuss the trade-offs of each option with the user to make a collaborative, informed decision.
