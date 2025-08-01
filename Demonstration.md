# Model Evaluation Commands

This document provides ready-to-use commands for evaluating all trained models in the project.

## Behavioral Cloning (BC) Models

### BC Needle Reach Model
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py --model-path models/bc_needle_reach_1752149760.zip --env NeedleReach-v0 --episodes 100
```

### BC Peg Transfer Model
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_bc.py --model-path models/bc_peg_transfer_1752155859.zip --env PegTransfer-v0 --episodes 100
```

## DAPG Models (Imitation-Augmented RL)

### DAPG Needle Reach Model
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py --model-path models/dapg_needle_reach_1751981351.zip --env-name NeedleReach-v0 --flatten
```

### DAPG Peg Transfer Model
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py --model-path models/dapg_peg_transfer_1752159090.zip --env-name PegTransfer-v0 --n-episodes 50 --flatten
```

## PPO Models (Pure RL)

### PPO Needle Reach Model
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/evaluate_rl.py --model-path models/ppo_needle_reach_1751965364.zip --env-name NeedleReach-v0 --flatten --dense-reward
```

## PPO Curriculum Learning Models

### PPO Level 1 Final Model (Curriculum Learning)
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/level1_final/model_level_1_final.zip --level 1  --render
```

### PPO Level 2 Final Model (Curriculum Learning)
```bash
docker compose -f docker/docker-compose.yml exec dvrk-dev python3 scripts/ppo_peg_transfer_curriculum/evaluate_curriculum_policy.py models/ppo_curriculum/runs/level2_final/model_level_2_final.zip --level 2  --render
```

## Notes

- **BC models**: Use `evaluate_bc.py` script
- **DAPG models**: Use `evaluate_rl.py` without `--dense-reward` flag (sparse rewards)
- **PPO models**: Use `evaluate_rl.py` with `--dense-reward` flag (dense rewards)
- All RL models require `--flatten` flag for observation compatibility
- Default episodes: BC (100), RL (10). Use `--episodes` or `--n-episodes` to customize
- Add `--no-render` flag to run without visualization for faster evaluation