import isaacgym
import isaacgymenvs

import torch
import torch.nn as nn
import gym, gymnasium
import copy
import itertools
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from algorithms.DRLR import DRLR, DRLR_DEFAULT_CONFIG
from algorithms.td3 import TD3, TD3_DEFAULT_CONFIG
# from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4

# Set random seed for reproducibility across PyTorch, NumPy, and Isaac Gym
RANDOM_SEED = 10 # 10, 11, 12 for reproducing the DRLR paper results
set_seed(RANDOM_SEED)

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================
# define RL agent: SAC (stochastic models for actor, and deterministic models for critic) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.action_layer = nn.Linear(256, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.linear_layer_3 = nn.Linear(256, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}

# Define IL agent, in DRLR, it is an offline TD3+BC agent (ref:https://github.com/sfujim/TD3_BC)
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions)
        )

    def compute(self, inputs, role):
        """Forward pass for deterministic action output."""
        return self.net(inputs["states"]), {}

# Alternative critic netwrok for trained offline TD3 (smaller network).
class Critic1(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        """Forward pass with state-action concatenation."""
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# ============================================================================
# Load and wrap the Isaac Gym environment
# ============================================================================
env = isaacgymenvs.make(seed=10,
                        task="FrankaCabinet",
                        num_envs=10,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=True)
env = wrap_env(env)
device = env.device


# ============================================================================
# # Instantiate replay and expert memory buffer
# ============================================================================
replay_buffer = RandomMemory(memory_size=350000, num_envs=env.num_envs, device=device, replacement=False)
expert_buffer = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=True)


# ============================================================================
# # Initiate an TD3 agent in order to load a trained offline TD3 agent, TODO can be optimized to only load the agent policy
# ============================================================================
models_td3 = {}
models_td3["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_td3["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_td3["critic_1"] = Critic1(env.observation_space, env.action_space, device)
models_td3["critic_2"] = Critic1(env.observation_space, env.action_space, device)
models_td3["target_critic_1"] = Critic1(env.observation_space, env.action_space, device)
models_td3["target_critic_2"] = Critic1(env.observation_space, env.action_space, device)
cfg_td3 = TD3_DEFAULT_CONFIG.copy()
agent_ref = TD3(models=models_td3,
                memory=replay_buffer,
                cfg=cfg_td3,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# load a trained offline TD3 agent,
agent_ref.load("./RefAgent/cab_td3_imperfect/checkpoints/agent_60000.pt")  # provided trained offline TD3 agent
models_ref = {}
models_ref["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
# Load the full saved state
# Extract just the policy weights
policy_state = models_td3["policy"].state_dict()
models_ref["policy"].load_state_dict(policy_state)
models_ref["policy"].eval()

########## enable this if using BC as refenrence agent
# ============================================================================
# # Initiate and Load BC works as the reference agent
# ============================================================================
# # Behavior clone (BC) requires 1 model to learn the expert behavior, directly load
# models_BC = {}
# models_BC["policy"] = ILActor(env.observation_space, env.action_space, device, clip_actions=True)
# # Load the full saved state
# saved_state = torch.load("./runs/BC-CAB-256128/checkpoints/agent_5000.pt")
# # Extract just the policy weights
# policy_state = saved_state["policy"]
# models_BC["policy"].load_state_dict(policy_state)
# models_BC["policy"].eval()


# ============================================================================
# # Initiate the RL agent: SAC
# ============================================================================
# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models_SAC = {}
models_SAC["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models_SAC["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_SAC["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_SAC["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_SAC["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models_SAC.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Create Emseabling Q networks (TODO RED-Q unfinished for DRLR, but it is not neccesary)
ensemble_size = 5
# Create ensemble of critics (each with unique parameters)
critics = []
target_critics = []
for i in range(ensemble_size):
    # Create new critic instance for each position
    critic = Critic(env.observation_space, env.action_space, device)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    critic.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    target_critic = copy.deepcopy(critic)  # Create target as deep copy

    critics.append(critic)
    target_critics.append(target_critic)
models_SAC["critics"] = critics
models_SAC["target_critics"] = target_critics


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
# configure and instantiate the agent
cfg = DRLR_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.99
cfg["batch_size"] = 128
cfg["random_timesteps"] = 0  # Add some random exploration at the start
cfg["learning_starts"] = 0   # Start learning after some experience
cfg["learn_entropy"] = True
cfg["grad_norm_clip"] = 1.0     # Add gradient clipping for stability
cfg["learning_rate"] = 3e-4     # Standard SAC learning rate
cfg["initial_entropy_value"] = 0.1     # Entropy learning rate
cfg["RED-Q_enable"] = False     #enable RED-Q
cfg["offline"] = False       # not important here
cfg["num_envs"] = env.num_envs
cfg["demo_file"] = "/home/chen/Downloads/new/memories/cab_imperfect.csv"
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 1000

agent = DRLR(models=models_SAC,
            models_il=models_ref,
            memory=replay_buffer,
            expert_memory=expert_buffer,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 350000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()