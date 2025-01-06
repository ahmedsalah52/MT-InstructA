import numpy as np
import torch
import torch.nn as nn



import torch.functional as F
from train_utils.backbones import *
from train_utils.necks.transformer import *
from train_utils.heads import *
from train_utils.models.base import arch
from collections import deque



import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        #assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            time_encoder     = nn.Embedding(config.max_episode_len, config.n_embd),
            state_encoder    = nn.Sequential(nn.Linear(config.state_size   ,config.n_embd),
                                            nn.LayerNorm(config.n_embd),
                                            nn.ReLU(),
                                            nn.Linear(config.n_embd,config.n_embd)),
            returns_encoder  = nn.Linear(1                   ,config.n_embd),
            actions_encoder  = nn.Linear(config.action_size  ,config.n_embd),
            commands_encoder = nn.Linear(config.command_size ,config.n_embd),
            hand_pos_encoder = nn.Linear(config.hand_pos_size,config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.step_len = config.step_len
        self.action_head = nn.Sequential(nn.Linear(config.n_embd, config.action_size, bias=True),
                                         nn.Tanh())
        
        self.reward_head = nn.Sequential(nn.Linear(config.n_embd, 1, bias=True),
                                         nn.Sigmoid())

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.time_encoder.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, commands,returns_to_go, states, hand_poses, actions,time_steps):
        b, t , _ = states.size()

        # encode time steps
        time_emb = self.transformer.time_encoder(time_steps)
        time_emb = self.transformer.drop(time_emb)

        #split state into cams
        #states = states.reshape(b,t,-1,self.config.state_size)
        #encode all inputs to emb_size and add time_emb
        returns_to_go = self.transformer.returns_encoder(returns_to_go) + time_emb
        states        = self.transformer.state_encoder(states)          + time_emb
        hand_poses    = self.transformer.hand_pos_encoder(hand_poses)   + time_emb
        actions       = self.transformer.actions_encoder(actions)       + time_emb
        commands      = self.transformer.commands_encoder(commands)

        
        stacked_sequence = torch.stack([returns_to_go, states, hand_poses, actions],dim=1).transpose(1,2)
        stacked_sequence = torch.flatten(stacked_sequence,start_dim=1,end_dim=2)
        stacked_sequence = torch.cat([commands,stacked_sequence],dim=1)
        
        stacked_sequence = self.transformer.ln_f(stacked_sequence)

        for block in self.transformer.h:
            stacked_sequence = block(stacked_sequence)

        #stacked_sequence[:,0] # out after the command (not used)
        stacked_sequence = stacked_sequence[:,1:].reshape(b,t,self.step_len,-1) # b , t , step_len , emb_dim
        # stacked_sequence = stacked_sequence.reshape(b,t,self.step_len,-1) # b , t , step_len , emb_dim in case no command
        stacked_sequence = stacked_sequence.transpose(1,2) # b , step_len , t , emb_dim

        actions_pred = self.action_head(stacked_sequence[:,-2]) # step  -> return, state , hand_pos, actions # we predict the actions after the hand_pos 
        rewards_pred = self.reward_head(stacked_sequence[:,-1]) # we predict the rewards after the actions
        return actions_pred,rewards_pred


    def configure_optimizers(self, weight_decay, learning_rate, cuda_device):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and cuda_device
        print(f"fused available: {fused_available} , cuda available: {cuda_device}")
        #extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,fused=use_fused)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


    @property
    def device(self):
        return next(self.parameters())
   

class DT_model(arch):
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
        self.reward_norm = args.reward_norm
        self.tasks = args.tasks
        @dataclass
        class GPTConfig:
            seq_len: int = args.seq_len
            step_len: int = 4
            block_size: int = (seq_len*step_len) + 1
            n_cams: int = len(args.cams)
            #vocab_size: int = None # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            max_episode_len: int = args.max_ep_len
            state_size: int = args.imgs_emps * len(args.cams)
            action_size: int = args.action_dim
            command_size: int = args.instuction_emps
            hand_pos_size: int = args.pos_dim
            n_layer: int = args.dt_n_layer
            n_head: int = args.dt_n_head
            n_embd: int = args.dt_embed_dim
            dropout: float = args.dt_dropout
            bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

        self.dt_model = GPT(GPTConfig())

    def reset_memory(self):
        args = self.args
        self.commands_embeddings = deque([], maxlen=1)  
        self.poses_embeddings    = deque([], maxlen=args.seq_len)  
        self.states_embeddings   = deque([], maxlen=args.seq_len)  
        self.actions             = deque([], maxlen=args.seq_len)  
        self.rewards             = deque([], maxlen=args.seq_len)  
        self.timesteps           = deque([], maxlen=args.seq_len)  
        self.attention_mask      = deque([], maxlen=args.seq_len)   
        self.eval_return_to_go   = 1.0

    def forward(self,batch):
        states_embeddings ,attention_mask,poses_embeddings= [],[],[]
        commands_to_use = None
        for i in range(self.args.seq_len):
            batch_step = {k: v[i].to(self.device) if type(v[i]) == torch.tensor else v[i] for k, v in batch.items()}
            states,commands,poses = self.backbone(batch_step,cat=False,vision=True,command=(i==0),pos=False)
            if i==0: commands_to_use = commands #update only the first command (to have one command per sequence)
            
            if self.neck:
                states,_,_ = self.neck((states,commands_to_use,poses),cat=False) 
            states_embeddings.append(states)
            poses_embeddings.append(batch_step['hand_pos'].to(torch.float32))
            attention_mask.append(batch_step['attention_mask'].to(torch.long))        
        #attention_mask = torch.tensor(batch['attention_mask'],dtype=torch.long).to(self.device)
        states_embeddings   = torch.stack(states_embeddings,dim=0).transpose(1,0).to(self.device)
        commands_embeddings = commands_to_use.unsqueeze(1)#torch.stack(commands_embeddings,dim=0).transpose(1,0).to(self.device)
        poses_embeddings    = torch.stack(poses_embeddings,dim=0).transpose(1,0).to(self.device)
        actions             = torch.stack(batch['action'],dim=0).transpose(1,0).to(self.device)
        timesteps           = torch.stack(batch['timesteps'],dim=0).transpose(1,0).to(self.device)
        returns_to_go       = torch.stack(batch[self.prompt],dim=0).unsqueeze(-1).transpose(1,0).to(torch.float32).to(self.device)
        attention_mask      = torch.stack(attention_mask,dim=0).transpose(1,0).to(self.device)
        #returns_to_go/= self.prompt_scale
        

        action_preds,rewards_preds = self.dt_model(
            commands= commands_embeddings,
            returns_to_go= returns_to_go,
            states=states_embeddings,
            hand_poses=poses_embeddings, 
            actions=actions,
            time_steps=timesteps
        )

        action_preds  = action_preds.transpose(1,0)
        rewards_preds = rewards_preds.transpose(1,0)
        return action_preds,rewards_preds
    
    def train_step(self,batch,device,opts=None):
        y_actions = torch.stack(batch['action'],dim=0).to(device)
        y_rewards = torch.stack(batch['reward'],dim=0).unsqueeze(-1).to(torch.float32).to(device)
        attention_mask = torch.stack(batch['attention_mask'],dim=0).unsqueeze(-1).to(torch.long).to(self.device)

        pred_actions,pred_rewards = self.forward(batch)
        
        actions_loss = self.masked_loss_fun(y_actions, pred_actions, attention_mask.repeat(1,1,4)) 
        rewards_loss = self.masked_loss_fun(y_rewards, pred_rewards, attention_mask)               

        return actions_loss + rewards_loss
    def masked_loss_fun(self,y_gt, y,mask):
        loss = (y_gt-y)**2
        # Apply the mask to ignore padded steps
        masked_loss = loss * mask
        # Calculate the average loss over the non-padded steps
        average_loss = torch.sum(masked_loss) / torch.sum(mask)
        return average_loss
    
    def eval_step(self,input_step):
        batch_step = {k : v.to(self.device) if type(v) == torch.tensor else v  for k,v in input_step.items()}
        states,commands,_ = self.backbone(batch_step,cat=False,vision=True,command=True,pos=False)
        if self.neck:
            states,commands,_ = self.neck((states,commands,None),cat=False)
        
        self.commands_embeddings.append(commands)
        self.poses_embeddings.append(batch_step['hand_pos'].to(torch.float32))
        self.states_embeddings.append(states.to(torch.float32))
        self.actions.append(torch.zeros_like(input_step['action'],dtype=torch.float32))
        self.timesteps.append(input_step['timesteps'].to(torch.long))
        self.rewards.append(torch.tensor([self.eval_return_to_go],dtype=torch.float32).to(self.device))
        self.attention_mask.append(torch.tensor([1],dtype=torch.long))
        
       

        states_embeddings   = torch.stack(list(self.states_embeddings)  ,dim=0).transpose(1,0).to(self.device)
        commands_embeddings = torch.stack(list(self.commands_embeddings),dim=0).transpose(1,0).to(self.device)
        poses_embeddings    = torch.stack(list(self.poses_embeddings)   ,dim=0).transpose(1,0).to(self.device)
        actions             = torch.stack(list(self.actions)            ,dim=0).transpose(1,0).to(self.device)
        timesteps           = torch.stack(list(self.timesteps)          ,dim=0).transpose(1,0).to(self.device)
        returns_to_go       = torch.stack(list(self.rewards)            ,dim=0).transpose(1,0).to(self.device)
        attention_mask      = torch.stack(list(self.attention_mask)     ,dim=0).transpose(1,0).to(self.device)

        if states_embeddings.shape[1]<self.args.seq_len:
            delta_seq_len     = self.args.seq_len - states_embeddings.shape[1]
            states_embeddings = torch.cat([states_embeddings ,torch.zeros((1 ,delta_seq_len, states_embeddings.shape[2]),dtype=torch.float32).to(self.device)],dim=1)
            poses_embeddings  = torch.cat([poses_embeddings  ,torch.zeros((1 ,delta_seq_len, poses_embeddings.shape[2]) ,dtype=torch.float32).to(self.device)],dim=1)
            actions           = torch.cat([actions           ,torch.zeros((1 ,delta_seq_len, actions.shape[2])          ,dtype=torch.float32).to(self.device)],dim=1)
            timesteps         = torch.cat([timesteps         ,torch.zeros((1 ,delta_seq_len)                            ,dtype=torch.long   ).to(self.device)],dim=1)
            returns_to_go     = torch.cat([returns_to_go     ,torch.zeros((1 ,delta_seq_len)                            ,dtype=torch.float32).to(self.device)],dim=1)
            attention_mask    = torch.cat([attention_mask    ,torch.zeros((1 ,delta_seq_len)                            ,dtype=torch.long   ).to(self.device)],dim=1)
        
        
        
        action_preds,rewards_preds = self.dt_model(
            commands= commands_embeddings,
            returns_to_go= returns_to_go.unsqueeze(-1),
            states=states_embeddings,
            hand_poses=poses_embeddings, 
            actions=actions,
            time_steps=timesteps
        )
        

        current_ts = input_step['timesteps'].item()
        current_ts = min(current_ts , self.args.seq_len-1)
       

        #if self.prompt != 'reward':
        if self.args.use_env_reward:
            task_name = self.tasks[input_step['task_id'].item()]
            self.eval_return_to_go -= input_step['reward']/self.dataset_specs['max_return_to_go'][task_name]
        else:
            if current_ts > 0:
                self.eval_return_to_go -= (rewards_preds[0,current_ts-1] * attention_mask[0,current_ts-1])
        self.actions[-1] = action_preds[:,current_ts]
        return action_preds[0,current_ts]
        
    def get_opt_params(self):
        params = self.backbone.get_opt_params() + [{"params": self.dt_model.parameters()}] 
        if self.neck:
            params += self.neck.get_opt_params()
       
        return params
    def get_optimizer(self):
        params = self.get_opt_params()
        #return self.dt_model.configure_optimizers(learning_rate=self.args.lr,weight_decay=self.args.weight_decay,cuda_device= 'cuda' in str(self.device))
        #return torch.optim.Adam(params,lr=self.args.lr)

        return torch.optim.AdamW(params,lr=self.args.lr,weight_decay=self.args.weight_decay)

   