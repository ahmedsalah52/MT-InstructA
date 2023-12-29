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

    
    train_metaenv = sequence_metaenv(train_tasks_commands,save_images=True,wandb_log = False,max_seq_len=1,train=True,cams_ids=[2,4])
    eval_metaenv  = sequence_metaenv(val_tasks_commands  ,save_images=True,wandb_log = False,max_seq_len=1,train=False,cams_ids=[2,4])
    #train_metaenv= Monitor(train_metaenv)
    #eval_metaenv = Monitor(train_metaenv)


    eval_callback = EvalCallback(eval_metaenv,# best_model_save_path=logs_dir+"/eval_logs/"+run_name,
                             #log_path=logs_dir+"/eval_logs/"+run_name,
                             deterministic=True,
                             #render=False,
                             eval_freq=10000,
                             n_eval_episodes=30)
    callbacks = CallbackList([ eval_callback])

    feature_extractor_kwargs = {"GM_args": args,
                                "features_dim": features_dim}
    # Create the PPO model with the custom policy
    model = PPO("MultiInputPolicy", train_metaenv, verbose=1,
                policy_kwargs=dict(share_features_extractor=True,
                                   features_extractor_class=genaral_model,
                                   features_extractor_kwargs=feature_extractor_kwargs,
                                   activation_fn=LeakyReLU,
                                   net_arch=[]),
                learning_rate=0.0001,
                batch_size=2)
    #print(model.policy)

    # Train the model
    model.learn(total_timesteps=2000000,callback=callbacks)
    # Save the trained model
    model.save("ppo_policy")

if __name__ == "__main__":
    main()