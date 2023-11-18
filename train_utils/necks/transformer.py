import torch.nn as nn
import torch
import math
from torch import nn, Tensor 
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        #self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
       
        x = x + self.pe[:x.size(0)].to(x.device)
        return self.dropout(x)
    
class transformer_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emp_size = args.emp_size
        self.pos_encoder  = PositionalEncoding(args.emp_size, dropout=args.neck_dropout, max_len=args.neck_max_len)
        encoder_layer     = nn.TransformerEncoderLayer(d_model=args.emp_size,dropout=args.neck_dropout, nhead=args.n_heads)
        self.encoder      = nn.TransformerEncoder(encoder_layer, num_layers=args.neck_layers)
        

    def forward(self, embeddings,cat=None):
        shape = embeddings.shape
        embeddings = embeddings.reshape(shape[0],-1,self.emp_size)
        embeddings = embeddings.permute(1,0,2)
        
        embeddings = embeddings * math.sqrt(self.emp_size)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.encoder(embeddings)
        return embeddings.permute(1,0,2).reshape(*shape)
    

      
    def get_opt_params(self):
        return  [
            {"params": self.encoder.parameters()}
             ]
