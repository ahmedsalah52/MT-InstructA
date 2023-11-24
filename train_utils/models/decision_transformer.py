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



class DecisionTransformer_multi(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            state_len,
            act_dim,
            command_dim,
            pos_emp,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.pos_emp = pos_emp
        self.command_dim = command_dim
        self.step_len = state_len + 3


        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_inner=self.step_len*hidden_size +1,
            **kwargs
        )
       
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_command = torch.nn.Linear(self.command_dim, hidden_size)
        self.embed_state = nn.ModuleList([torch.nn.Linear(self.state_dim, hidden_size) for _ in range(state_len)]) 
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_pos = torch.nn.Linear(self.pos_emp, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        #self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_reward = torch.nn.Linear(hidden_size, 1)
        #self.command_preds = torch.nn.Linear(hidden_size, command_dim)

    def forward(self, states, actions, poses, command,returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        command = command[:,0:1] #only first command

        # embed each modality with a different head
        state_embeddings = [state_emp(states) for i,state_emp in enumerate(self.embed_state)]
        action_embeddings = self.embed_action(actions)
        command_embeddings = self.embed_command(command)
        time_embeddings = self.embed_timestep(timesteps)
        pos_embeddings  = self.embed_pos(poses)
        returns_embeddings  = self.embed_return(returns_to_go)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = [state + time_embeddings for state in state_embeddings]
        action_embeddings = action_embeddings + time_embeddings
        #command_embeddings = command_embeddings + time_embeddings
        pos_embeddings  = pos_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings,pos_embeddings, *state_embeddings,action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.step_len*seq_length, self.hidden_size)
        stacked_inputs = torch.cat((command_embeddings,stacked_inputs), dim=1)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            [attention_mask]*self.step_len, dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.step_len*seq_length)
        stacked_attention_mask = torch.cat((stacked_attention_mask, torch.ones((batch_size, 1), dtype=torch.long).to(stacked_attention_mask.device)), dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0) pos (1), command (2), state (3) , action (4); i.e. x[:,1,t] is the token for s_t
        x = x[:,1:].reshape(batch_size, seq_length, self.step_len, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
       
        reward_preds = self.predict_reward(x[:,-1])  
        #state_preds  = self.predict_state(x[:,2])    
        action_preds = self.predict_action(x[:,-2])   # predict next action given state

        return action_preds,reward_preds

    

class DecisionTransformer_2cams(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            state_len,
            act_dim,
            command_dim,
            pos_emp,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.pos_emp = pos_emp
        self.command_dim = command_dim
        self.step_len = state_len + 2
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_command = torch.nn.Linear(self.command_dim, hidden_size)
        self.embed_state = nn.ModuleList([torch.nn.Linear(self.state_dim, hidden_size) for _ in range(state_len)]) 
        #self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        #self.embed_pos = torch.nn.Linear(self.pos_emp, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        #self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        #self.command_preds = torch.nn.Linear(hidden_size, command_dim)

    def forward(self, states, actions, pos_embeddings, command,returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = [state_emp(states[:,:,i,:]) for i,state_emp in enumerate(self.embed_state)]
        #action_embeddings = self.embed_action(actions)
        command_embeddings = self.embed_command(command)
        time_embeddings = self.embed_timestep(timesteps)
        #pos_embeddings  = self.embed_pos(poses)
        returns_embeddings  = self.embed_return(returns_to_go)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = [state + time_embeddings for state in state_embeddings]
        #action_embeddings = action_embeddings + time_embeddings
        #command_embeddings = command_embeddings + time_embeddings
        pos_embeddings  = pos_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings,pos_embeddings, *state_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.step_len*seq_length, self.hidden_size)
        
        #add command 
        stacked_inputs = torch.cat((command_embeddings,stacked_inputs), dim=1)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            [attention_mask]*self.step_len, dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.step_len*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        
        #add command mask
        stacked_attention_mask = torch.cat((stacked_attention_mask, torch.ones((batch_size, 1), dtype=torch.long).to(stacked_attention_mask.device)), dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
       
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0) pos (1), command (2), state (3) , action (4); i.e. x[:,1,t] is the token for s_t
        x = x[:,1:].reshape(batch_size, seq_length, self.step_len, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        #command_preds = self.command_preds(x[:,0])  
        #state_preds  = self.predict_state(x[:,2])    
        action_preds = self.predict_action(x[:,-1])   # predict next action given state

        return action_preds

    


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
            action_tanh=False,
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

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        #self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        #self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

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

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        #return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        #state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return action_preds


class DL_model(arch):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.args = args
        self.loss_fun  = self.loss_funs[args.loss_fun](args)
        self.backbone  = self.backbones[args.backbone](args)
        self.neck = self.necks[args.neck](args)
        self.flatten = nn.Flatten()
        self.preprocess_image = self.backbone.preprocess_image
        self.prompt = args.prompt
        self.prompt_scale = args.prompt_scale 
        self.dummy_param = nn.Parameter(torch.zeros(0))

        self.dl_model = DecisionTransformer_multi(
            state_dim=len(args.cams)*args.imgs_emps,
            state_len=1,#len(args.cams),
            act_dim=args.action_dim,
            command_dim=args.instuction_emps,
            pos_emp=args.pos_emp,
            max_length=args.seq_len,
            max_ep_len=args.max_ep_len,
            hidden_size=args.dt_embed_dim,
            n_layer=args.dt_n_layer,
            n_head=args.dt_n_head,
            activation_function=args.dt_activation_function,
            n_positions=1024,
            resid_pdrop=args.dt_dropout,
            attn_pdrop=args.dt_dropout,
        )
       
    def reset_memory(self):
        args = self.args
        self.states_embeddings   = deque([torch.zeros(1, len(args.cams)*args.imgs_emps).to(self.dummy_param.device) for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.commands_embeddings = deque([torch.zeros(1, args.instuction_emps).to(self.dummy_param.device) for _ in range(args.seq_len)], maxlen=1)  
        self.poses_embeddings    = deque([torch.zeros(1, args.pos_emp).to(self.dummy_param.device)    for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.actions             = deque([torch.zeros(1, args.action_dim).to(self.dummy_param.device) for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.timesteps           = deque([torch.zeros(1  ,dtype=torch.int).to(self.dummy_param.device)    for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.rewards             = deque([torch.zeros(1  ,dtype=torch.float16).to(self.dummy_param.device)    for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.attention_mask      = deque([torch.zeros(1  ,dtype=torch.int).to(self.dummy_param.device)    for _ in range(args.seq_len)], maxlen=args.seq_len)  
        self.eval_return_to_go   = self.args.target_rtg/self.prompt_scale

    def forward(self,batch):
        states_embeddings ,commands_embeddings,poses_embeddings= [],[],[]
        for i in range(self.args.seq_len):
            batch_step = {}
            for k,vs in batch.items():
                batch_step[k] = vs[i]
           
            batch_step = {k : v.to(self.dummy_param.device) if k != 'instruction' else v  for k,v in batch_step.items()}
            
            states,commands,poses = self.backbone(batch_step,cat=False)
            if self.neck:
                states,commands,poses = self.neck((states,commands,poses),cat=False)
            states_embeddings.append(states)
            poses_embeddings.append(poses)

            commands_embeddings.append(commands)
        

        states_embeddings   = torch.stack(states_embeddings,dim=0).transpose(1,0).to(self.dummy_param.device)
        commands_embeddings = torch.stack(commands_embeddings,dim=0).transpose(1,0).to(self.dummy_param.device)
        poses_embeddings    = torch.stack(poses_embeddings,dim=0).transpose(1,0).to(self.dummy_param.device)
        actions             = torch.stack(batch['action'],dim=0).transpose(1,0).to(self.dummy_param.device)
        timesteps           = torch.stack(batch['timesteps'],dim=0).transpose(1,0).to(self.dummy_param.device)
        returns_to_go       = torch.stack(batch[self.prompt],dim=0).unsqueeze(-1).transpose(1,0).float().to(self.dummy_param.device)
        returns_to_go/= self.prompt_scale

        
        batch_size,seq_length,_ = actions.shape
        attention_mask = torch.ones((batch_size, self.args.seq_len), dtype=torch.long).to(self.dummy_param.device)

        action_preds,rewards_preds = self.dl_model.forward(
            states_embeddings,
            actions,
            poses_embeddings,
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
        states,commands,poses = self.backbone(batch_step,cat=False)
        if self.neck:
            states,commands,poses = self.neck((states,commands,poses),cat=False)
        
        self.states_embeddings.append(states)
        self.commands_embeddings.append(commands)
        self.poses_embeddings.append(poses)
        
        self.actions[-1] = input_step['action']
        self.actions.append(torch.zeros_like(input_step['action']))

        self.timesteps.append(input_step['timesteps'])
        self.rewards.append(torch.tensor([1.0],dtype=torch.float).to(self.dummy_param.device))
        self.attention_mask.append(torch.tensor([1]))
        

        states_embeddings   = torch.stack([s.to(self.dummy_param.device) for s in self.states_embeddings],dim=0).transpose(1,0).to(self.dummy_param.device)
        commands_embeddings = torch.stack([s.to(self.dummy_param.device) for s in self.commands_embeddings],dim=0).transpose(1,0).to(self.dummy_param.device)
        poses_embeddings    = torch.stack([s.to(self.dummy_param.device) for s in self.poses_embeddings],dim=0).transpose(1,0).to(self.dummy_param.device)
        actions             = torch.stack([s.to(self.dummy_param.device) for s in self.actions],dim=0).transpose(1,0).to(self.dummy_param.device)
        timesteps           = torch.stack([s.to(self.dummy_param.device) for s in self.timesteps],dim=0).transpose(1,0).to(self.dummy_param.device)
        returns_to_go       = torch.stack([s.to(self.dummy_param.device) for s in self.rewards],dim=0).transpose(1,0).to(self.dummy_param.device)
        attention_mask      = torch.stack([s.to(self.dummy_param.device) for s in self.attention_mask],dim=0).transpose(1,0).to(self.dummy_param.device)

        action_preds,rewards_preds = self.dl_model.forward(
        states_embeddings,
        actions,
        poses_embeddings,
        commands_embeddings,
        returns_to_go.unsqueeze(-1),
        timesteps, 
        attention_mask=attention_mask
        )
        

        if self.prompt != 'reward':
            if self.args.use_env_reward:
                self.eval_return_to_go -= input_step['reward']/self.prompt_scale
            else:
                self.eval_return_to_go -= (rewards_preds[0,-2] * attention_mask[0,-2])
        print(f'reward {input_step["reward"]} , predicted reward {rewards_preds[0,-2] * self.prompt_scale}')
        return action_preds[0,-1]
        
    def get_opt_params(self):
        params = self.backbone.get_opt_params() + [{"params": self.dl_model.parameters()}]
       
        return params
    def get_optimizer(self):
        params = self.get_opt_params()

        return torch.optim.Adam(params,lr=self.args.lr)

   