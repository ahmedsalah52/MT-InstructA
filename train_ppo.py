from typing import Dict, List, Tuple, Type, Union
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy,BaseModel
from stable_baselines3 import PPO
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch import Tensor, nn
import torch.nn as nn
import torch as th
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

class backbone(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)
class actor(nn.Module):
    def __init__(self, in_dim, out_dim,backbone):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)
    
class critic(nn.Module):
    def __init__(self, in_dim, out_dim,backbone):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)
class My_MlpExtractor(MlpExtractor):
    def __init__(self, feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto") -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        self.backbone    = backbone(4,feature_dim)
        self.value_net   = critic(feature_dim,feature_dim,self.backbone)
        self.policy_net  = actor(feature_dim,feature_dim,self.backbone)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent_pi, latent_vf = super().forward(features)
        return latent_pi, latent_vf
    
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        print(self)
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return super().forward(obs, deterministic=deterministic)
    

    def _build_mlp_extractor(self):
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = My_MlpExtractor(feature_dim=120,net_arch=[],activation_fn=nn.ReLU)
       
# Create a vectorized environment
vec_env = make_vec_env("CartPole-v1", n_envs=4)
print(vec_env.action_space.n)
# Instantiate your custom policy

# Create the PPO model with the custom policy
model = PPO(CustomPolicy, vec_env, verbose=1)

# Train the model
model.learn(total_timesteps=25000)

# Save the trained model
model.save("ppo_cartpole_custom_policy")
