# DRLR


## General info
Project is created with:
* Isaac Gym Preview 4
* SKRL

## Installation requirements
1. Install Isaac gym and Example RL Environments

https://developer.nvidia.com/isaac-gym

https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

For my project, I Install it in a new conda environment.

2. Install Deep reinforcemenr learning lib(SKRL)

https://skrl.readthedocs.io/en/latest/intro/installation.html

	
## Setup
Before run this project:

```
$ cd ../repository
$ conda activate rlgpu
$ export LD_LIBRARY_PATH=/home/chen/anaconda3/envs/rlgpu/lib:$LD_LIBRARY_PATHA
```
## How to use Deep Reinforcement Learning with Reference policy(DRLR)
The default training task is open drawer (https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/franka_cabinet.py)

1. Generate and collect demonstrations. If you have human demonstrations, skip this step.
```
$ python task_ppo.py   # train PPO to complete task with good enough performance. After training, eval policy and collect 'expert' demos.
```
2. Train an IL policy, load your demo file in the BC config
```
$ python task_BC.py   # train PPO to complete task with good enough performance. After training, eval policy and collect 'expert' demos.
```
3. Run training script with DRLR, load your demo file and trained BC agent in the config.

DRLR algorithm(https://arxiv.org/abs/2509.04069)
```
$ python task_DRLR.py
```
