# Autonomous Driving RL Policy — Reward Design & Environment (L5Kit)

## Context
This code was developed as part of my graduate research on reinforcement-learning-based 
autonomous vehicle policy training, built on top of Woven Planet/Lyft's open-source 
L5Kit simulation framework (https://github.com/woven-planet/l5kit). L5Kit provides the 
base dataset handling, map API, and closed-loop simulation infrastructure; the files in 
this repo are the components I designed and implemented within that framework.

**Note:** This repo contains selected files extracted from a larger private research 
codebase for demonstration purposes. It is not a runnable standalone project — it 
requires the full L5Kit environment, a licensed dataset, and lab-specific configuration 
to execute.

## What's original vs. framework-provided

| File | Description | Authorship |
|---|---|---|
| `reward_vec.py` | Custom multi-term reward function — physics-based target speed (braking-distance kinematics), map-based lane/crosswalk/traffic-light queries, neighbor-distance filtering | ~98% original design and implementation |
| `l5_env.py` | Gym-compatible simulation environment — observation space, action rescaling, reward integration | Implements the standard OpenAI Gym interface (`reset`/`step`); internal logic (observation construction, action handling, reward wiring) is original |
| `policy_train_eval.py` | PPO training pipeline setup — custom feature extractor, hyperparameter scheduling, vectorized parallel environments | Original training configuration and setup |
| `evaluation1.py` | Custom evaluation harness — multi-metric scoring (distance, speed, lane adherence) for trained policies | Original evaluation design |

## Key technical highlights
- Physics-based reward shaping using kinematic braking-distance equations
- Custom multi-metric evaluation framework (not just reward — separate interpretable metrics)
- Parallelized training using fork-based vectorized environments on Linux
- Debugged cross-device (CPU/GPU/MPS) inference issues in a multi-module PyTorch policy

## Background
This work is part of my thesis implementation (https://github.com/adilce009/egoDriveSafe/tree/main) for studying safe, human-like driving policies for closed-loop simulation, evaluated on the Lyft Level 5 Prediction Dataset.
