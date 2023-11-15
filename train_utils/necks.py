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
        self.att_head_emp = args.att_head_emp
        self.pos_encoder  = PositionalEncoding(args.att_head_emp, dropout=args.neck_dropout, max_len=args.neck_max_len)
        encoder_layer     = nn.TransformerEncoderLayer(d_model=args.att_head_emp,dropout=args.neck_dropout, nhead=args.n_heads)
        self.encoder      = nn.TransformerEncoder(encoder_layer, num_layers=args.neck_layers)
        

    def forward(self, embeddings):
        shape = embeddings.shape
        embeddings = embeddings.reshape(shape[0],-1,self.att_head_emp)
        embeddings = embeddings.permute(1,0,2)
        
        embeddings = embeddings * math.sqrt(self.att_head_emp)
        embeddings = self.pos_encoder(embeddings)
        embeddings = self.encoder(embeddings)
        return embeddings.permute(1,0,2).reshape(*shape)
    

      
    def get_opt_params(self):
        return  [
            {"params": self.encoder.parameters()}
             ]



class CrossAttentionLayer(nn.Module):
    def __init__(self, q_dim,cross_attention_dim,embed_dim, num_heads=1,dropout=0.2):
        super(CrossAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim 
        total_embed_dim = num_heads * embed_dim
        # Linear transformations for queries, keys, and values
        self.query_linear = nn.Linear(q_dim, total_embed_dim)
        self.key_linear = nn.Linear(cross_attention_dim, total_embed_dim)
        self.value_linear = nn.Linear(cross_attention_dim, total_embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Output linear layer
        self.out_linear = nn.Linear(total_embed_dim,cross_attention_dim)

    def forward(self, x, conditional_x):
        batch_size, len_x, _     = x.size()
        _, len_conditional_x, _ = conditional_x.size()

        # Linear transformations
        query = self.query_linear(x)
        key   = self.key_linear(conditional_x)
        value = self.value_linear(conditional_x)

        # Reshape for multi-head attention
        query = query.view(batch_size, len_x            , self.num_heads, self.head_dim).transpose(1, 2)
        key   =   key.view(batch_size, len_conditional_x, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, len_conditional_x, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)

        # Reshape and concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, len_x, -1)

        # Apply output linear layer
        output = self.out_linear(attended_values)
        output = self.dropout(output)
        output = output + x
        return output

class CrossAttentionTransformerMultiLayer(nn.Module):
    def __init__(self, num_layers,q_dim,cross_attention_dim,embed_dim, num_heads=1,dropout=0.2):
        super(CrossAttentionTransformerMultiLayer, self).__init__()

        self.layers = nn.ModuleList([
            CrossAttentionLayer(q_dim=q_dim, cross_attention_dim=cross_attention_dim,embed_dim=embed_dim, num_heads=num_heads,dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self,x, conditional_x):
        for layer in self.layers:
            x = layer(x, conditional_x)
        return x

class CrossAttentionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.att_head_emp = args.att_head_emp
       
        self.encoder = CrossAttentionTransformerMultiLayer(num_layers=args.neck_layers,
                                                                q_dim=args.att_head_emp,
                                                                cross_attention_dim=args.att_head_emp,
                                                                embed_dim=args.att_head_emp,
                                                                num_heads=args.n_heads,
                                                                dropout=args.neck_dropout)
        
        self.flatten = nn.Flatten()
    def forward(self, input_x):
        images_emps,text_emps,pos_emps = input_x
        shape = images_emps.shape

        images_emps = images_emps.reshape(shape[0],-1,self.att_head_emp)
        text_emps   = text_emps.reshape(shape[0],-1,self.att_head_emp)
        images_emps  = self.encoder(images_emps,text_emps)
        images_emps = images_emps
        text_emps   = text_emps

        text_images_embeddings = torch.cat([images_emps,text_emps],dim=1)
        text_images_embeddings = self.flatten(text_images_embeddings)
        return torch.cat([text_images_embeddings,pos_emps],dim=1)
         
         
    

      
    def get_opt_params(self):
        return  [
            {"params": self.encoder.parameters()}
             ]

