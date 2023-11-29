import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import torch.functional as F
from train_utils.backbones import *
from train_utils.necks.transformer import *
from train_utils.heads import *
from train_utils.models.base import arch
from collections import deque



class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_tasks = torch.nn.Linear(hidden_size, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, tasks, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        tasks_embeddings = self.embed_tasks(tasks)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)


        stacked_inputs         = torch.cat([tasks_embeddings,stacked_inputs],dim=1)
        stacked_attention_mask = torch.cat([torch.ones(batch_size,1).to(attention_mask.device),stacked_attention_mask],dim=1)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x[:,1:,:].reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return  action_preds, return_preds

   

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
      
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds[0,-1]
    
class DL_model_obs(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun](args)
        #self.backbone  = self.backbones[args.backbone](args)
        #self.neck = self.necks[args.neck](args)
        self.flatten = nn.Flatten()
        self.preprocess_image = None #self.backbone.preprocess_image
        self.prompt = args.prompt
        self.prompt_scale = args.prompt_scale 
        self.dummy_param = nn.Parameter(torch.zeros(0))
        self.state_dim = 39
        self.dl_model = DecisionTransformer(
            state_dim=self.state_dim,
            state_len=1,#len(args.cams),
            act_dim=args.action_dim,
            command_dim=args.instuction_emps,
            pos_emp=args.pos_emp,
            max_length=args.seq_len,
            max_ep_len=args.max_ep_len,
            hidden_size=args.dt_embed_dim,
            n_inner=args.dt_embed_dim*4,
            n_layer=args.dt_n_layer,
            n_head=args.dt_n_head,
            activation_function=args.dt_activation_function,
            n_positions=1024,
            resid_pdrop=args.dt_dropout,
            attn_pdrop=args.dt_dropout,
        )
        self.task_embeddings = nn.Embedding(len(args.tasks), args.dt_embed_dim)
        self.states_mean = 0.18576333177347915
        self.states_std = 0.3379336491547313
       
    def reset_memory(self):
        args = self.args
        self.commands_embeddings = deque([torch.zeros(1, args.instuction_emps).to(self.dummy_param.device) for _ in range(args.seq_len)], maxlen=1)  
        #self.poses_embeddings    = deque([torch.zeros(1, args.pos_emp).to(self.dummy_param.device)    for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.states_embeddings   = deque([], maxlen=args.seq_len)  
        self.actions             = deque([], maxlen=args.seq_len)  
        self.rewards             = deque([], maxlen=args.seq_len)  
        self.timesteps           = deque([], maxlen=args.seq_len)  
        self.attention_mask      = deque([], maxlen=args.seq_len)  
        self.eval_return_to_go   = self.args.target_rtg/self.prompt_scale
        
        
         
    def forward(self,batch):
        """states_embeddings ,commands_embeddings,poses_embeddings= [],[],[]
        for i in range(self.args.seq_len):
            batch_step = {}
            for k,vs in batch.items():
                batch_step[k] = vs[i]
           
            batch_step = {k : v.to(self.dummy_param.device) if k != 'instruction' else v  for k,v in batch_step.items()}
            
            _,commands,_ = self.backbone(batch_step,cat=False,vision=False,command=True,pos=False)
         
            states_embeddings.append(batch_step['obs'])
            #poses_embeddings.append(poses)

            commands_embeddings.append(commands)
        """
        
        #tasks_id = self.task_embeddings(batch['task_id'][:,0,:]) # use only the first task id from each sequence, (it doesn't change throught the sequence)
        
        states_embeddings   = torch.stack(batch['obs'],dim=0).transpose(1,0).to(self.dummy_param.device)
        #commands_embeddings = torch.stack(batch['task_id'][0],dim=0).to(self.dummy_param.device)
        commands_embeddings = self.task_embeddings(batch['task_id'][0].to(self.device))#.transpose(1,0)

        #poses_embeddings    = torch.stack(poses_embeddings,dim=0).transpose(1,0).to(self.dummy_param.device)
        actions             = torch.stack(batch['action'],dim=0).transpose(1,0).to(self.dummy_param.device)
        timesteps           = torch.stack(batch['timesteps'],dim=0).transpose(1,0).to(self.dummy_param.device)
        returns_to_go       = torch.stack(batch[self.prompt],dim=0).unsqueeze(-1).transpose(1,0).float().to(self.dummy_param.device)
        returns_to_go/= self.prompt_scale
        states_embeddings = (states_embeddings - self.states_mean) / self.states_std
        
        batch_size,seq_length,_ = actions.shape
        attention_mask = torch.ones((batch_size, self.args.seq_len), dtype=torch.long).to(self.dummy_param.device)
        #print(commands_embeddings.shape)
        #print(actions.shape)
        action_preds,rewards_preds = self.dl_model.forward(
            states_embeddings,
            actions,
            commands_embeddings,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
        )

        action_preds = action_preds.transpose(1,0)
        rewards_preds = rewards_preds.transpose(1,0).squeeze(-1)
        return action_preds,rewards_preds
    
    def train_step(self,batch,device,opts=None):
        y_actions = torch.stack(batch['action'],dim=0).to(device)
        y_rewards = torch.stack(batch['reward'],dim=0).float().to(device)/self.prompt_scale

        pred_actions,pred_rewards = self.forward(batch)
        
        actions_loss = self.loss_fun(pred_actions, y_actions)
        rewards_loss = self.loss_fun(pred_rewards, y_rewards)

        return actions_loss + rewards_loss
    
    def eval_step(self,input_step):
        batch_step = {k : v.to(self.dummy_param.device) if k != 'instruction' else v  for k,v in input_step.items()}
        #_,commands,_ = self.backbone(batch_step,cat=False,vision=False,command=True,pos=False)
       
        
        states = batch_step['obs']
        
        self.commands_embeddings.append(self.task_embeddings(batch_step['task_id']))
        #self.poses_embeddings.append(poses)
        self.states_embeddings.append(states.to(torch.float32))
        self.actions.append(torch.zeros_like(input_step['action'],dtype=torch.float32))
        self.timesteps.append(input_step['timesteps'].to(torch.long))
        self.rewards.append(torch.tensor([self.eval_return_to_go],dtype=torch.float32).to(self.device))
        self.attention_mask.append(torch.tensor([1],dtype=torch.long))
        
       

        states_embeddings   = torch.stack(list(self.states_embeddings),dim=0).transpose(1,0).to(self.device)
        ids_embeddings      = torch.stack(list(self.commands_embeddings),dim=0).transpose(1,0).to(self.device)
        #poses_embeddings    = torch.stack([s.to(self.dummy_param.device) for s in self.poses_embeddings],dim=0).transpose(1,0).to(self.dummy_param.device)
        actions             = torch.stack(list(self.actions)       ,dim=0).transpose(1,0).to(self.device)
        timesteps           = torch.stack(list(self.timesteps)     ,dim=0).transpose(1,0).to(self.device)
        returns_to_go       = torch.stack(list(self.rewards)       ,dim=0).transpose(1,0).to(self.device)
        attention_mask      = torch.stack(list(self.attention_mask),dim=0).transpose(1,0).to(self.device)

        states_embeddings = (states_embeddings - self.states_mean) / self.states_std

        #action =  self.dl_model.get_action(states_embeddings.squeeze(0),actions.squeeze(0),None,returns_to_go.squeeze(0),timesteps)
        #self.actions[-1] = action.unsqueeze(0)
        #self.eval_return_to_go -= input_step['reward']/self.prompt_scale
        #return action
        if False and states_embeddings.shape[1]<self.args.seq_len:
            delta_seq_len     = self.args.seq_len - states_embeddings.shape[1]
            states_embeddings = torch.cat([torch.zeros((1 ,delta_seq_len,states_embeddings.shape[2]),dtype=torch.float32).to(self.device),states_embeddings],dim=1)
            actions           = torch.cat([torch.zeros((1 ,delta_seq_len,actions.shape[2]),dtype=torch.float32).to(self.device)          ,actions          ],dim=1)
            timesteps         = torch.cat([torch.zeros((1 ,delta_seq_len),dtype=torch.long).to(self.device)                              ,timesteps        ],dim=1)
            ids_embeddings    = torch.cat([torch.zeros((1 ,delta_seq_len),dtype=torch.long).to(self.device)                              ,ids_embeddings   ],dim=1)
            returns_to_go     = torch.cat([torch.zeros((1 ,delta_seq_len),dtype=torch.float32).to(self.device)                           ,returns_to_go    ],dim=1)
            attention_mask    = torch.cat([torch.zeros((1 ,delta_seq_len),dtype=torch.long).to(self.device)                              ,attention_mask   ],dim=1)
        
        #print(ids_embeddings.shape)
        #print(actions.shape)
        action_preds,rewards_preds = self.dl_model.forward(
        states_embeddings,
        actions,
        ids_embeddings,
        returns_to_go.unsqueeze(-1),
        timesteps, 
        attention_mask=attention_mask
        )
        

        if self.prompt != 'reward':
            if self.args.use_env_reward:
                self.eval_return_to_go -= input_step['reward']/self.prompt_scale
            else:
                self.eval_return_to_go -= (rewards_preds[0,-2] * attention_mask[0,-2])
        self.actions[-1] = action_preds[:,-1]
        return action_preds[0,-1]
        
    def get_opt_params(self):
        params =  [{"params": self.dl_model.parameters()}] #+self.backbone.get_opt_params() 
       
        return params
    def get_optimizer(self):
        params = self.get_opt_params()

        return torch.optim.Adam(params,lr=self.args.lr)

   