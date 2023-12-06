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

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, custom_actor_kwargs, custom_critic_kwargs, **kwargs):
        super().__init__(**kwargs)
        self.custom_actor_kwargs = custom_actor_kwargs
        self.custom_critic_kwargs = custom_critic_kwargs
        self._build_custom_networks()

    def _build_custom_networks(self):
        # Override _build_mlp_extractor to create custom actor and critic
        self._build_custom_actor()
        self._build_custom_critic()

    def _build_custom_actor(self):
        # Define your custom actor network here
        # Example: a simple MLP with one hidden layer
        self.custom_actor = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n),
        )

    def _build_custom_critic(self):
        # Define your custom critic network here
        # Example: a simple MLP with one hidden layer
        self.custom_critic = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _build_mlp_extractor(self):
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = self.features_extractor
        self._build_custom_actor()
        self._build_custom_critic()

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        # Use custom actor for action distribution
        mean_actions = self.custom_actor(latent_pi)

        if isinstance(self.action_dist, Distribution):
            return self.action_dist.proba_distribution(mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.custom_critic(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

class fc(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        print('fc',x.shape)
        return self.fc(x)
class My_MlpExtractor(MlpExtractor):
    def __init__(self, feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        action_dim: int = 1) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        self.value_net  = fc(feature_dim,1)
        self.policy_net = fc(feature_dim,action_dim)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        print('Custom_MlpExtractor',features.shape)
        latent_pi, latent_vf = super().forward(features)
        return latent_pi, latent_vf
    
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        print('CustomPolicy',obs.shape)
        return super().forward(obs, deterministic=deterministic)
    

    def _build_mlp_extractor(self):
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = My_MlpExtractor(feature_dim=self.features_dim,net_arch=[64, 64],activation_fn=nn.ReLU,action_dim=self.action_space.n)
       

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
