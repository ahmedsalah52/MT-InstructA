#
import math
import torch
import torch.nn as nn

class LoRA(nn.Module):

    def __init__(self, length=2, pool_size =1,n_layer=1,embed_dim=128,top_k=1, dropout_rate=0.0, init_prompts="zeros",
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
        if  mod_q: 
            length -= 1
        if  mod_v:
            length -= 1
        if mod_k: 
            length += 1
        if mod_ff: 
            # mlp is 4 * embed_dim
            length += 4
        
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
        # lora_a_batched: [n_layer x length x batch_size x rank x embed_dim]
        # lora_b_batched: [n_layer x length x batch_size x embed_dim x rank]
        lora_a_batched = self.lora_a[idx].permute(1,2,0,3,4)
        lora_b_batched = self.lora_b[idx].permute(1,2,0,3,4)
        

        return lora_a_batched,lora_b_batched
    

    def add_dropout(self, batched_prompt):
        return batched_prompt