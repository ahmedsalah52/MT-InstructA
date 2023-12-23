#
import math
import torch
import torch.nn as nn

class LoRA(nn.Module):

    def __init__(self, length=0, pool_size =1,n_layer=1,embed_dim=128,top_k=1, dropout_rate=0.0, init_prompts="zeros",
                 rank=4, mod_q=True, mod_v=True, mod_k=False, mod_ff=True, 
                 lora_alpha=None, log_mod_stats=False, **kwargs):
        
        super().__init__()
        self.log_mod_stats = log_mod_stats
        self.rank = rank
        self.mod_v = mod_v
        self.mod_q = mod_q
        self.mod_k = mod_k
        self.mod_ff = mod_ff
        self.lora_alpha = lora_alpha if lora_alpha is not None else self.rank * 2
        self._scaling = self.lora_alpha / self.rank
        
        length = 7
        
        self.length = length
        self.pool_size = pool_size
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self._setup_prompt()

    @property
    def scaling(self):
        return self._scaling
        
    def _setup_prompt(self):
        self.lora_a = nn.Parameter(torch.zeros((self.pool_size, self.n_layer, self.length, self.embed_dim, self.rank)))
        self.lora_b = nn.Parameter(torch.zeros((self.pool_size, self.n_layer, self.length, self.rank, self.embed_dim)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
    
    def extract_prompt(self, idx):
        """
        Args:
            idx: torch.Tensor. Indices to lookup.

        """
        # idx: [batch_size x 1]
        # lora_a_batched: [n_layer x length x batch_size x rank      x embed_dim]
        # lora_b_batched: [n_layer x length x batch_size x embed_dim x rank]
        lora_a_batched = self.lora_a[idx].permute(1,2,0,3,4)
        lora_b_batched = self.lora_b[idx].permute(1,2,0,3,4)
        lora_params = []
        for a,b in zip(lora_a_batched,lora_b_batched):
            qa,qb = a[0],b[0]
            va,vb = a[1],b[1]
            ffa1,ffb1 = a[2:3],b[2:6]
            ffa2,ffb2 = a[3:7],b[6:7]

            #permute mlp layers
            ffa1 = ffa1.permute(1,2,0,3).flatten(start_dim=1,end_dim=2)
            ffa2 = ffa2.permute(1,2,0,3).flatten(start_dim=1,end_dim=2)
            ffb1 = ffb1.permute(1,2,0,3).flatten(start_dim=2,end_dim=3)
            ffb2 = ffb2.permute(1,2,0,3).flatten(start_dim=2,end_dim=3)
            

            lora_params.append({
                'q_ab' : torch.matmul(qa,qb),
                'v_ab' : torch.matmul(va,vb),
                'ff1_ab': torch.matmul(ffa1,ffb1),
                'ff2_ab': torch.matmul(ffa2,ffb2)

            })
            
        return lora_params
    

    def add_dropout(self, batched_prompt):
        return batched_prompt
    


class LoRA_layer(nn.Module):
    def __init__(self,pool_size=1,in_embed_dim=128,out_embed_dim=128,rank=4,lora_alpha=None):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else self.rank * 2
        self._scaling = self.lora_alpha / self.rank
        self.pool_size = pool_size
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self._setup_prompt()

    @property
    def scaling(self):
        return self._scaling
        
    def _setup_prompt(self):
        self.lora_a = nn.Parameter(torch.zeros((self.pool_size, self.in_embed_dim , self.rank)))
        self.lora_b = nn.Parameter(torch.zeros((self.pool_size, self.rank         , self.out_embed_dim)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self,x,idx):
        a,b = self.lora_a[idx],self.lora_b[idx]
        ab = torch.matmul(a,b)
        return  (torch.matmul(x,ab) * self.scaling)
