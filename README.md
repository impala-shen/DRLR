# DRLR
This is the code for implementing DRLR algorithm in paper: Solving Robotics Tasks with Prior Demonstration via Exploration-Efficient Deep Reinforcement Learning (https://arxiv.org/abs/2509.04069), which is under review for Journal Frontiers in Robotics and AI.
If you find it useful, please consider cite this paper.

## General info
Project is created with:
* Isaac Gym Preview 4
* SKRL 1.4.0

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
The default training task is open drawer in sparse reward setting, to reproduce the same results, remember to change the distance reward gain to 0 in the task config (https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/franka_cabinet.py)

1. Collect demonstrations. 3 demostrations are provided in the Demos folder, you can also try generate 'fake' expert demo yourself by training PPO.
```
$ python task_ppo.py   # train PPO to complete task with good enough performance. After training, eval policy and collect 'expert' demos.
```
2. Train a Ref policy, load the demo file in the BC/TD3+BC config. You can also directly use Ref policies in the RefAgent folder.
```
$ python task_BC.py   # train Ref policy to complete task.
```
3. Run training script with DRLR, load your demo file and trained Ref agent in the config.

DRLR algorithm(https://arxiv.org/abs/2509.04069)
```
$ python task_DRLR.py
```
