# Expert Demonstrations

This directory should contain expert demonstration files for training.

## Required Files:

- `expert_demo_needle_reach.pkl` - Expert demonstrations for Needle Reach task
- `expert_demo_peg_transfer.pkl` - Expert demonstrations for Peg Transfer task

## Generating Demonstrations:

Use the generation scripts to create expert demonstrations:

```bash
# Generate Needle Reach demonstrations
python3 generate_expert_data_needle_reach.py

# Generate Peg Transfer demonstrations  
python3 generate_expert_data_peg_transfer.py
```

## File Format:

Demonstration files should be pickle files containing a list of trajectories, where each trajectory is a dictionary with:

- `obs` or `observations`: Observation sequences
- `acts` or `actions`: Action sequences

The observations can be either:
- Dictionary format with `observation`, `achieved_goal`, `desired_goal` keys
- Flattened arrays (concatenated from dictionary components)