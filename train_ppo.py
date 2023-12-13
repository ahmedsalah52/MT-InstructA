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
from stable_baselines3.common.torch_layers import MlpExtractor
from collections import deque
from meta_env import sequence_metaenv
from train_utils.metaworld_dataset import split_dict
from train_utils.args import  parser ,process_args
import json 


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
        return x , x
class actor(nn.Module):
    def __init__(self, in_dim, out_dim,backbone):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
    
class critic(nn.Module):
    def __init__(self, in_dim, out_dim,backbone):
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
        self.backbone    = backbone(39,feature_dim)
        self.value_net   = critic(feature_dim,feature_dim,self.backbone)
        self.policy_net  = actor(feature_dim,feature_dim,self.backbone)
        self.max_seq_len = 10
        self.reset_seq_buffer()

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]: 
        latent_pi, latent_vf = super().forward(features)
        return latent_pi, latent_vf
    

    def reset_seq_buffer(self):
        self.obs_list = deque([], maxlen=self.max_seq_len)
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    def forward(self, obs, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return super().forward(obs, deterministic=deterministic)
    def extract_features(self,obs):
        obs_ = obs['obs'].to(torch.float32)
        ret = self.mlp_extractor.backbone(obs_)
        return ret


    def _build_mlp_extractor(self):
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = My_MlpExtractor(feature_dim=64,net_arch=[],activation_fn=nn.ReLU)
    
    def predict_values(self, obs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        obs_ = obs['obs'].to(torch.float32)
        latent_pi , latent_vf = self.mlp_extractor.backbone(obs_)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

def main():
    args = parser.parse_args()
    args = process_args(args)
    tasks_commands = json.load(open(args.tasks_commands_dir))
    tasks_commands = {k:list(set(tasks_commands[k])) for k in args.tasks} #the commands dict should have the same order as args.tasks list
    train_tasks_commands,val_tasks_commands = split_dict(tasks_commands,args.commands_split_ratio,seed=args.seed)

    train_metaenv = sequence_metaenv(train_tasks_commands,save_images=False,wandb_log = False,max_seq_len=10)
   
    # Create the PPO model with the custom policy
    model = PPO(CustomPolicy, train_metaenv, verbose=1,
                policy_kwargs=dict(share_features_extractor=False),
                learning_rate=0.0003,
                batch_size=64)

    # Train the model
    model.learn(total_timesteps=2000000)

    # Save the trained model
    model.save("ppo_cartpole_custom_policy")

if __name__ == "__main__":
    main()