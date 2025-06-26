# Active Context: Project Roadmap and Strategic Focus

## 1. Current Focus: Implement `PegTransfer` Task
Our primary focus is to replicate our successful training workflow on a more complex task to prove its generality. Based on a thorough analysis of the original SurRoL codebase, we have established a clear task hierarchy and made a strategic decision on the next step.

## 2. Surgical Task Hierarchy

We have categorized the available PSM-based tasks into three distinct difficulty tiers:

### Tier 1: Basic Reaching (Completed)
-   **Task**: `NeedleReach`
-   **Core Skill**: 3D space navigation.
-   **Status**: **100% Complete**. This validated our baseline environment and learning workflow.

### Tier 2: Single-Arm Pick-and-Place (Next Target)
These tasks introduce object interaction and require a more sophisticated, waypoint-based expert policy.
-   `NeedlePick`: Basic grasp and lift.
-   `GauzeRetrieve`: Grasp, lift, and move.
-   `PegTransfer`: Grasp, lift, move, and high-precision placement.

### Tier 3: Bimanual Coordination (Future Scope)
This is the most complex tier, requiring significant framework extensions to support dual-arm control.
-   `BiPegTransfer`: Involves two arms and object hand-offs.
-   **Status**: **Out of scope for now**. To be considered only if time permits after mastering Tier 2.

## 3. Strategic Decision: Tackle `PegTransfer` Next

After careful consideration, we have decided to **bypass the simpler Tier 2 tasks and directly target `PegTransfer`**.

-   **Rationale**: The expert policies for all Tier 2 tasks are based on the same waypoint controller logic. `PegTransfer` uses the most complete, 6-stage version of this logic. By solving for `PegTransfer`, we effectively solve the core technical challenges of `NeedlePick` and `GauzeRetrieve` simultaneously. This "high-risk, high-reward" approach is the most efficient path to validating our framework's capability for complex manipulation.

## 4. Immediate Actions
The next concrete step is to begin the implementation of the `PegTransfer` environment.
1.  Create the new environment file: `src/dvrk_gym/envs/peg_transfer.py`.
2.  Implement the environment logic, referencing `archive/SurRoL/surrol/tasks/peg_transfer.py`.
3.  Adapt `scripts/generate_expert_data.py` to include a `peg-transfer` mode with the 6-waypoint expert policy.
