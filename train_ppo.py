from typing import Dict, List, Tuple, Type, Union
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy,BaseModel
import gymnasium as gym

from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_util import make_vec_env
from torch import Tensor, nn
import torch.nn as nn
import torch as th
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor,BaseFeaturesExtractor
from collections import deque
from meta_env import sequence_metaenv
from train_utils.metaworld_dataset import split_dict
from train_utils.args import  parser ,process_args
import json 
from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList ,EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
from train_utils.RL_model import genaral_model
class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__(negative_slope, inplace)


class backbone(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        #self.fc = nn.Linear(in_dim, out_dim)
        #self.gru = nn.GRU(out_dim, out_dim,num_layers=2,batch_first=True)
        self.network = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(negative_slope= 0.01),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(negative_slope= 0.01),
        )
    def forward(self, x):
        x = x[:,-1,:]
        x = self.network(x)
        #x , h = self.gru(x)
        return x 
class actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
    
class critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)
class My_MlpExtractor(MlpExtractor):
    def __init__(self, feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto") -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        self.value_net   = critic(feature_dim,feature_dim)
        self.policy_net  = actor(feature_dim,feature_dim)

    def forward(self, obs) -> Tuple[th.Tensor, th.Tensor]: 
        #obs_ = obs['obs'].to(torch.float32)
        #obs_ = self.backbone(obs_)
        latent_pi, latent_vf = self.policy_net(obs), self.value_net(obs)
        return latent_pi, latent_vf
    


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def forward(self, obs, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return super().forward(obs, deterministic=deterministic)

"""
    def _build_mlp_extractor(self):
      
        self.mlp_extractor = My_MlpExtractor(feature_dim=64,net_arch=[],activation_fn=nn.ReLU)
    
    def get_distribution(self, obs) -> Distribution:
        obs_ = obs['obs'].to(torch.float32)
        obs_ = self.mlp_extractor.backbone(obs_)
        latent_pi = self.mlp_extractor.forward_critic(obs_)
        return self._get_action_dist_from_latent(latent_pi)
    def predict_values(self, obs) -> th.Tensor:
        obs_ = obs['obs'].to(torch.float32)
        obs_ = self.mlp_extractor.backbone(obs_)
        latent_vf = self.mlp_extractor.forward_critic(obs_)
        return self.value_net(latent_vf)"""

class My_Feature_extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        observation_dim = observation_space['obs'].shape[-1]
        self.backbone    = backbone(observation_dim,features_dim)

    def forward(self, observations):
        observations = observations['obs'].to(torch.float32)
        observations = self.backbone(observations)
        return observations



class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

def main():
    args = parser.parse_args()
    args = process_args(args)
    features_dim = 256

    tasks_commands = json.load(open(args.tasks_commands_dir))
    tasks_commands = {k:list(set(tasks_commands[k])) for k in args.tasks} #the commands dict should have the same order as args.tasks list
    train_tasks_commands,val_tasks_commands = split_dict(tasks_commands,args.commands_split_ratio,seed=args.seed)

    train_metaenv = sequence_metaenv(train_tasks_commands,save_images=False,wandb_log = False,max_seq_len=10,train=True)
    eval_metaenv  = sequence_metaenv(val_tasks_commands  ,save_images=False,wandb_log = False,max_seq_len=10,train=False)
    #train_metaenv= Monitor(train_metaenv)
    #eval_metaenv = Monitor(train_metaenv)


    eval_callback = EvalCallback(train_metaenv,# best_model_save_path=logs_dir+"/eval_logs/"+run_name,
                             #log_path=logs_dir+"/eval_logs/"+run_name,
                             deterministic=True,
                             #render=False,
                             eval_freq=10000,
                             n_eval_episodes=30)
    callbacks = CallbackList([ eval_callback])

    feature_extractor_kwargs = {"features_dim": features_dim}
    # Create the PPO model with the custom policy
    model = PPO("MultiInputPolicy", train_metaenv, verbose=1,
                policy_kwargs=dict(share_features_extractor=False,
                                   features_extractor_class=My_Feature_extractor,
                                   features_extractor_kwargs=feature_extractor_kwargs,
                                   activation_fn=LeakyReLU,
                                   net_arch=[]),
                learning_rate=0.0001,
                batch_size=256)
    #print(model.policy)

    # Train the model
    model.learn(total_timesteps=2000000,callback=callbacks)
    # Save the trained model
    model.save("ppo_policy")

if __name__ == "__main__":
    main()